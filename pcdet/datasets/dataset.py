from collections import defaultdict
from pathlib import Path

import numpy as np
import torch.utils.data as torch_data

from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder
from ..utils.center_utils import draw_umich_gaussian, gaussian_radius


class DatasetTemplate(torch_data.Dataset):
    def __init__(self,
                 dataset_cfg=None,
                 class_names=None,
                 training=True,
                 root_path=None,
                 logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(
            self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE,
                                          dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range)
        self.data_augmentor = DataAugmentor(
            self.root_path,
            self.dataset_cfg.DATA_AUGMENTOR,
            self.class_names,
            logger=self.logger) if self.training else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR,
            point_cloud_range=self.point_cloud_range,
            training=self.training)

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False
        self.feature_map_stride = dataset_cfg.get('FEATURE_MAP_STRIDE', None)

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict,
                                  pred_dicts,
                                  class_names,
                                  output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array(
                [n in self.class_names for n in data_dict['gt_names']],
                dtype=np.bool_)

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict, 'gt_boxes_mask': gt_boxes_mask
                })

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(
                data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in data_dict['gt_names']],
                dtype=np.int32)
            gt_boxes = np.concatenate(
                (data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(
                    np.float32)),
                axis=1)
            data_dict['gt_boxes'] = gt_boxes
            if self.training and self.dataset_cfg.USE_HEATMAP:
                self.compute_heatmap(gt_boxes, gt_classes, data_dict)

        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(data_dict=data_dict)

        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)

        return data_dict

    def compute_heatmap(self, gt_boxes, gt_classes, data_dict):

        max_objs = self.dataset_cfg.MAX_OBJ
        tasks = self.dataset_cfg.TASKS
        class_names_by_task = [t['HEAD_CLS_NAME'] for t in tasks]

        feature_map_size = self.grid_size[:2] // self.feature_map_stride

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in class_names_by_task:
            # print("classes: ", gt_dict["gt_classes"], "name", class_name)
            task_masks.append([
                np.where(gt_classes == class_name.index(i) + 1 + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            task_name = []
            for m in mask:
                task_box.append(gt_boxes[m])
                task_class.append(gt_classes[m] - flag2)
            task_boxes.append(np.concatenate(task_box, axis=0))
            task_classes.append(np.concatenate(task_class))
            flag2 += len(mask)
        '''
        for task_box in task_boxes:
            # limit rad to [-pi, pi]
            task_box[:, -1] = box_np_ops.limit_period(
                task_box[:, -1], offset=0.5, period=np.pi * 2
            )
        '''
        # print(gt_dict.keys())

        hms, anno_boxs, inds, masks, cats = [], [], [], [], []

        for idx, task in enumerate(tasks):
            hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[1],
                           feature_map_size[0]),
                          dtype=np.float32)
            # [reg, hei, dim, rots, rotc]
            anno_box = np.zeros((max_objs, 8), dtype=np.float32)

            ind = np.zeros((max_objs), dtype=np.int64)
            mask = np.zeros((max_objs), dtype=np.uint8)
            cat = np.zeros((max_objs), dtype=np.int64)
            direction = np.zeros((max_objs), dtype=np.int64)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                w, l, h = task_boxes[idx][k][3], task_boxes[idx][k][4], \
                          task_boxes[idx][k][5]
                w, l = w / self.voxel_size[
                    0] / self.feature_map_stride, l / self.voxel_size[
                        1] / self.feature_map_stride
                if w > 0 and l > 0:
                    radius = gaussian_radius(
                        (l, w), min_overlap=self.dataset_cfg.GAUSSIAN_OVERLAP)
                    radius = max(self.dataset_cfg.MIN_RADIUS, int(radius))

                    # be really careful for the coordinate system of your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][1], \
                              task_boxes[idx][k][2]

                    coor_x, coor_y = (x - self.point_cloud_range[0]) / self.voxel_size[0] / self.feature_map_stride, \
                                     (y - self.point_cloud_range[1]) / self.voxel_size[1] / self.feature_map_stride

                    ct = np.array([coor_x, coor_y], dtype=np.float32)
                    ct_int = ct.astype(np.int32)

                    # throw out not in range objects to avoid out of array area when creating the heatmap
                    if not (0 <= ct_int[0] < feature_map_size[0]
                            and 0 <= ct_int[1] < feature_map_size[1]):
                        continue

                    draw_umich_gaussian(hm[cls_id], ct, radius)

                    new_idx = k
                    x, y = ct_int[0], ct_int[1]

                    if not (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1]):
                        # a double check, should never happen
                        print(x, y, y * feature_map_size[0] + x)
                        assert False

                    cat[new_idx] = cls_id
                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1

                    rot = task_boxes[idx][k][-1]
                    anno_box[new_idx] = np.concatenate(
                        (ct - (x, y), z, np.log(task_boxes[idx][k][3:6]),
                         np.sin(rot), np.cos(rot)),
                        axis=None)

            hms.append(hm)
            anno_boxs.append(anno_box)
            masks.append(mask)
            inds.append(ind)
            cats.append(cat)

        data_dict.update({
            'hm': hms,
            'anno_box': anno_boxs,
            'ind': inds,
            'mask': masks,
            'cat': cats
        })

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)),
                                          mode='constant',
                                          constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros(
                        (batch_size, max_gt, val[0].shape[-1]),
                        dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ["hm", "anno_box", "ind", "mask", "cat"]:
                    ret[key] = defaultdict(list)
                    res = []
                    for elem in val:
                        for idx, ele in enumerate(elem):
                            ret[key][str(idx)].append(ele)
                    for kk, vv in ret[key].items():
                        res.append(np.stack(vv, axis=0))
                    ret[key] = np.array(res)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
