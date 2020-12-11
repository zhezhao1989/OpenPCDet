import numpy as np
import torch
import torch.nn as nn
import copy
from ...utils import loss_utils
from ...utils.center_utils import ddd_decode
from ..model_utils.weight_init import kaiming_init

class SeparateHead(nn.Module):
    def __init__(self, heads, input_channels, num_middle_filter, init_bias=-2.19):
        super().__init__()

        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = input_channels
            for i in range(num_conv - 1):
                conv_layers.extend([
                    nn.Conv2d(
                        c_in, num_middle_filter,
                        kernel_size=3, stride=1, padding=1, bias=True
                    ),
                    nn.BatchNorm2d(num_middle_filter),
                    nn.ReLU()
                ])
                c_in = num_middle_filter
            conv_layers.append(nn.Conv2d(
                num_middle_filter, classes,
                kernel_size=3, stride=1, padding=1,
                bias = True
            ))
            conv_layers = nn.Sequential(*conv_layers)
            self.__setattr__(head, conv_layers)

        self.init_weights()

    def init_weights(self):
        for head in self.heads:
            if head == 'hm':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)
            else:
                for m in self.__getattr__(head).modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

    def forward(self, x):

        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict

class CenterHead(nn.Module):
    """CenterHead for CenterPoint """

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training, voxel_size, feature_map_stride, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.feature_map_stride = feature_map_stride
        self.pc_range = point_cloud_range
        self.predict_boxes_when_training = predict_boxes_when_training

        shared_conv_num_filter = self.model_cfg.SHARED_CONV_NUM_FILTER
        self.shared_conv = nn.Sequential(
            nn.Conv2d(input_channels, shared_conv_num_filter, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(shared_conv_num_filter, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.tasks = nn.ModuleList()
        num_heatmap_convs = 2
        for task in self.model_cfg.TASKS:
            heads = copy.deepcopy(self.model_cfg.COMMON_HEADS)
            heads.update(dict(hm=(len(task['HEAD_CLS_NAME']), num_heatmap_convs)))
            seperate_head = SeparateHead(heads, shared_conv_num_filter, self.model_cfg.NUM_MIDDLE_FILTER)
            self.tasks.append(seperate_head)

        self.class_names = class_names
        self.crit = loss_utils.FocalLoss()
        self.crit_reg = loss_utils.RegL1Loss()
        self.code_weights = self.model_cfg.LOSS_WEIGHTS['code_weights']
        self.forward_ret_dict = {}

    def forward(self, data_dict):
        x = data_dict['spatial_features_2d']

        ret_dicts = []
        x = self.shared_conv(x)

        for task in self.tasks:
            ret_dicts.append(task(x))

        self.forward_ret_dict['preds_dict'] = ret_dicts
        self.forward_ret_dict['data_dict'] = data_dict

        if not self.training or self.predict_boxes_when_training:
            final_pred_scores, final_pred_labels, final_pred_boxes = self.generate_predicted_boxes(ret_dicts)

            data_dict['final_pred_boxes'] = final_pred_boxes
            data_dict['final_pred_labels'] = final_pred_labels
            data_dict['final_pred_scores'] = final_pred_scores

        return data_dict

    def generate_predicted_boxes(self, preds_dicts):

        final_pred_boxes = []
        final_pred_scores = []
        final_pred_labels = []

        multihead_label_mapping = []
        for idx in range(len(preds_dicts)):
            head_label_indices = torch.from_numpy(np.array([
                    self.class_names.index(cur_name) + 1 for cur_name in self.model_cfg.TASKS[idx]['HEAD_CLS_NAME']
                    ]))
            multihead_label_mapping.append(head_label_indices)

        for task_id, preds_dict in enumerate(preds_dicts):
            batch_hm = preds_dict['hm'].sigmoid_()
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']
            batch_dim = torch.exp(preds_dict['dim'])
            batch_rots = preds_dict['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict['rot'][:, 1].unsqueeze(1)

            temp = ddd_decode(
                batch_hm,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                None,
                None,
                None,
                reg=batch_reg,
                K=self.model_cfg.K,
                score_threshold=None,#self.model_cfg.SCORE_THRESH,
                voxel_size=self.voxel_size,
                pc_range=self.pc_range,
                feature_map_stride=self.feature_map_stride
            )
            final_pred_labels.append(multihead_label_mapping[task_id][temp['pred_labels'].long()])
            final_pred_scores.append(temp['pred_scores'])
            final_pred_boxes.append(temp['pred_boxes'])

        final_pred_boxes = torch.cat(final_pred_boxes, dim=1)
        final_pred_labels = torch.cat(final_pred_labels, dim=1)
        final_pred_scores = torch.cat(final_pred_scores, dim=1)
        return final_pred_scores, final_pred_labels, final_pred_boxes

    def clip_sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y

    def get_loss(self):

        preds_dicts = self.forward_ret_dict['preds_dict']
        data_dict = self.forward_ret_dict['data_dict']

        loss = 0
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict['hm'] = self.clip_sigmoid(preds_dict['hm'])

            hm_loss = self.crit(preds_dict['hm'], data_dict['hm'][task_id])

            target_box = data_dict['anno_box'][task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                preds_dict['rot']), dim=1)

            # Regression loss for dimension, offset, height, rotation
            box_loss = self.crit_reg(preds_dict['anno_box'], data_dict['mask'][task_id], data_dict['ind'][task_id].long(), target_box)

            loc_loss = (box_loss*box_loss.new_tensor(self.code_weights)).sum()

            loss += hm_loss + loc_loss

        tb_dict = {
            'loss': loss.item()
        }
        return loss, tb_dict
