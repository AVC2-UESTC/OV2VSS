import os
import os.path as osp
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose

import random
from PIL import Image

import torch

@DATASETS.register_module()
class CustomDataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
                         [i]] = summary_table_data[1][i] / 100.0
        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results

@DATASETS.register_module()
class CustomDataset2(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations2(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def load_annotations2(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            # for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
            #     img_info = dict(filename=img)
            #     if ann_dir is not None:
            #         seg_map = img.replace(img_suffix, seg_map_suffix)
            #         img_info['ann'] = dict(seg_map=seg_map)
            #     img_infos.append(img_info)
            for ann_name in mmcv.scandir(ann_dir, seg_map_suffix, recursive=True):
                img=ann_name.replace(seg_map_suffix,img_suffix)

                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = ann_name
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
                # print(img_info)
                # exit()

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        # print(len(self.CLASSES))
        # exit()
        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
                         [i]] = summary_table_data[1][i] / 100.0
        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results



@DATASETS.register_module()
class CustomDataset_cityscape_clips(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 dilation=[-9,-6,-3],
                 istraining=True):
        # self.pipeline = Compose(pipeline)
        if istraining:
            self.pipeline_load = Compose(pipeline[:2])
            self.pipeline_process = Compose(pipeline[2:])
        else:
            self.pipeline_load = Compose(pipeline[:1])
            self.pipeline_process = Compose(pipeline[1:])

        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations2(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)
        print(len(self.img_infos))
        self.flip_video=True
        print("flip video: ",self.flip_video)
        self.dilation=dilation

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def load_annotations2(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            # for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
            #     img_info = dict(filename=img)
            #     if ann_dir is not None:
            #         seg_map = img.replace(img_suffix, seg_map_suffix)
            #         img_info['ann'] = dict(seg_map=seg_map)
            #     img_infos.append(img_info)
            for ann_name in mmcv.scandir(ann_dir, seg_map_suffix, recursive=True):
                # print(ann_name)
                img=ann_name.replace(seg_map_suffix,img_suffix)

                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = ann_name
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        # img_infos = img_infos[:100]
        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        # exit()
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)

        if self.flip_video:
            # print("here")
            if random.random()<0.5:
                # imglist=imglist[::-1]
                dilation_used=[-i for i in self.dilation]
            else:
                dilation_used=self.dilation

        # print(img_info, ann_info)
        # exit()
        try:
            img_anns=[]
            for ii in dilation_used:
                img_info_one={}
                filename=img_info['filename']
                seg_map=img_info['ann']['seg_map']
                value_i_splits=filename.split('_')
                im_name_new = "_".join(
                    value_i_splits[:-2] + [(str(int(value_i_splits[-2]) + ii)).rjust(6, "0")] + value_i_splits[-1:])
                # value_i_splits=seg_map.split('_')
                # seg_map_new = "_".join(
                #     value_i_splits[:-2] + [(str(int(value_i_splits[-2]) - ii)).rjust(6, "0")] + value_i_splits[-1:])

                img_info_one['filename']=im_name_new
                img_info_one['ann']=dict(seg_map=seg_map)
                ann_info_one=img_info_one['ann']
                img_anns.append([img_info_one, ann_info_one])
                
                if not os.path.isfile(self.img_dir+'/'+im_name_new):
                    print('not here',self.img_dir+'/'+im_name_new)
                    assert False
        except:
            dilation_used=[-i for i in dilation_used]
            img_anns=[]
            for ii in dilation_used:
                img_info_one={}
                filename=img_info['filename']
                seg_map=img_info['ann']['seg_map']
                value_i_splits=filename.split('_')
                im_name_new = "_".join(
                    value_i_splits[:-2] + [(str(int(value_i_splits[-2]) + ii)).rjust(6, "0")] + value_i_splits[-1:])
                # value_i_splits=seg_map.split('_')
                # seg_map_new = "_".join(
                #     value_i_splits[:-2] + [(str(int(value_i_splits[-2]) - ii)).rjust(6, "0")] + value_i_splits[-1:])

                img_info_one['filename']=im_name_new
                img_info_one['ann']=dict(seg_map=seg_map)
                ann_info_one=img_info_one['ann']
                img_anns.append([img_info_one, ann_info_one])
                if not os.path.isfile(self.img_dir+'/'+im_name_new):
                    assert False
        img_anns.append([img_info, ann_info])

        # print(img_anns)

        clips_img = []
        clips_target=[]
        clips_meta=[]
        results_all=[]

        img_info_clips, ann_info_clips, seg_fields_clips, img_prefix_clips, seg_prefix_clips, filename_clips=[],[],[],[],[],[]
        ori_filename_clips, img_clips, img_shape_clips, ori_shape_clips, pad_shape_clips=[],[],[],[],[]
        scale_factor_clips, img_norm_cfg_clips, gt_semantic_seg_clips=[],[],[]

        for kkk in img_anns:
            results=dict(img_info=kkk[0], ann_info=kkk[1])
            self.pre_pipeline(results)
            self.pipeline_load(results)
            results_all.append(results)
            img_info_clips.append(results['img_info'])
            ann_info_clips.append(results['ann_info'])
            seg_fields_clips.append(results["seg_fields"])
            img_prefix_clips.append(results["img_prefix"])
            seg_prefix_clips.append(results["seg_prefix"])
            filename_clips.append(results["filename"])
            ori_filename_clips.append(results["ori_filename"])
            img_clips.append(results["img"]) 
            img_shape_clips.append(results["img_shape"])
            ori_shape_clips.append(results["ori_shape"])
            pad_shape_clips.append(results["pad_shape"])
            scale_factor_clips.append(results["scale_factor"])
            img_norm_cfg_clips.append(results["img_norm_cfg"])
            gt_semantic_seg_clips.append(results["gt_semantic_seg"])

        results_new=dict(img_info=img_info_clips[-1],ann_info=ann_info_clips[-1],seg_fields=seg_fields_clips[-1],
            img_prefix=img_prefix_clips[-1],seg_prefix=seg_prefix_clips[-1],
            filename=filename_clips[-1],ori_filename=ori_filename_clips[-1],img=img_clips,
            img_shape=img_shape_clips[-1],ori_shape=ori_shape_clips[-1],
            pad_shape=pad_shape_clips[-1],scale_factor=scale_factor_clips[-1],
            img_norm_cfg=img_norm_cfg_clips[-1],gt_semantic_seg=gt_semantic_seg_clips)

        return self.pipeline_process(results_new)

        # img_info = self.img_infos[idx]
        # ann_info = self.get_ann_info(idx)
        # results = dict(img_info=img_info, ann_info=ann_info)
        # self.pre_pipeline(results)
        # return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """
        img_info = self.img_infos[idx]
        dilation_used=self.dilation
        img_anns=[]
        for ii in dilation_used:
            img_info_one={}
            filename=img_info['filename']
            seg_map=img_info['ann']['seg_map']
            value_i_splits=filename.split('_')
            im_name_new = "_".join(
                value_i_splits[:-2] + [(str(int(value_i_splits[-2]) + ii)).rjust(6, "0")] + value_i_splits[-1:])
            # value_i_splits=seg_map.split('_')
            # seg_map_new = "_".join(
            #     value_i_splits[:-2] + [(str(int(value_i_splits[-2]) - ii)).rjust(6, "0")] + value_i_splits[-1:])

            img_info_one['filename']=im_name_new
            img_info_one['ann']=dict(seg_map=seg_map)
            ann_info_one=img_info_one['ann']
            img_anns.append([img_info_one, ann_info_one])
            # print(self.img_dir+'/'+im_name_new)
            if not os.path.isfile(self.img_dir+'/'+im_name_new):
                assert False
        img_anns.append([img_info, img_info['ann']])

        clips_img = []
        clips_target=[]
        clips_meta=[]
        results_all=[]

        img_info_clips, ann_info_clips, seg_fields_clips, img_prefix_clips, seg_prefix_clips, filename_clips=[],[],[],[],[],[]
        ori_filename_clips, img_clips, img_shape_clips, ori_shape_clips, pad_shape_clips=[],[],[],[],[]
        scale_factor_clips, img_norm_cfg_clips, gt_semantic_seg_clips=[],[],[]
        for kkk in img_anns:
            results = dict(img_info=kkk[0], ann_info=kkk[1])
            self.pre_pipeline(results)
            self.pipeline_load(results)
            results_all.append(results)
            img_info_clips.append(results['img_info'])
            ann_info_clips.append(results['ann_info'])
            seg_fields_clips.append(results["seg_fields"])
            img_prefix_clips.append(results["img_prefix"])
            seg_prefix_clips.append(results["seg_prefix"])
            filename_clips.append(results["filename"])
            ori_filename_clips.append(results["ori_filename"])
            img_clips.append(results["img"]) 
            img_shape_clips.append(results["img_shape"])
            ori_shape_clips.append(results["ori_shape"])
            pad_shape_clips.append(results["pad_shape"])
            scale_factor_clips.append(results["scale_factor"])
            img_norm_cfg_clips.append(results["img_norm_cfg"])

        results_new=dict(img_info=img_info_clips[-1],ann_info=ann_info_clips[-1],seg_fields=seg_fields_clips[-1],
            img_prefix=img_prefix_clips[-1],seg_prefix=seg_prefix_clips[-1],
            filename=filename_clips[-1],ori_filename=ori_filename_clips[-1],img=img_clips,
            img_shape=img_shape_clips[-1],ori_shape=ori_shape_clips[-1],
            pad_shape=pad_shape_clips[-1],scale_factor=scale_factor_clips[-1],
            img_norm_cfg=img_norm_cfg_clips[-1])

        return self.pipeline_process(results_new)

        # img_info = self.img_infos[idx]
        # results = dict(img_info=img_info)
        # self.pre_pipeline(results)
        # return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
                         [i]] = summary_table_data[1][i] / 100.0
        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results



@DATASETS.register_module()
class CustomDataset_video(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── video
        │   │   │   ├── origin
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── mask
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 dilation=None,
                 clipnum=None):
        self.pipeline = Compose(pipeline)
        self.img_dir = ''
        self.img_suffix = img_suffix
        self.ann_dir = ''
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        self.dilation=dilation
        self.clipnum=clipnum

        # join paths if data_root is specified
        # if self.data_root is not None:
        #     if not osp.isabs(self.img_dir):
        #         self.img_dir = osp.join(self.data_root, self.img_dir)
        #     if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
        #         self.ann_dir = osp.join(self.data_root, self.ann_dir)
        #     if not (self.split is None or osp.isabs(self.split)):
        #         self.split = osp.join(self.data_root, self.split)

        with open(os.path.join(self.data_root,self.split+'.txt')) as f:
            lines=f.readlines()
            self.videolists = [line[:-1] for line in lines]
        
        self.imgdic={}
        self.img_all=[]
        for video in self.videolists:
            v_path = os.path.join(self.data_root,'data',video,'origin')
            imglist = sorted(os.listdir(v_path))
            self.imgdic[video]=imglist
            self.img_all=self.img_all+[[video, img] for img in imglist]

        # print(self.reduce_zero_label)
        # exit()
        # if self.split=='train':
        #     self.img_all=self.img_all[:100]
        # load annotations
        # self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
        #                                        self.ann_dir,
        #                                        self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        # return len(self.img_infos)
        # return len(self.videolists)
        return len(self.img_all)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results, img_dir,  ann_dir):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = img_dir
        results['seg_prefix'] = ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        # video  = self.videolists[idx]
        # imglist = self.imgdic[video]
        # imglist_s = imglist[-self.dilation[0]:]
        # if len(imglist_s)<1:
        #     return None
        # idx = np.random.choice(list(range(len(imglist_s))))-self.dilation[0]
        # this_step=[]
        # for dil in self.dilation:
        #     this_step.append(idx+dil)
        # this_step.append(idx)

        # clips_img = []
        # clips_target=[]
        # clips_meta=[]
        # for i in this_step:
        #     img_name=imglist[i]
        #     img_info=dict(filename=img_name)
        #     seg_map = img_name.replace(img_suffix, seg_map_suffix)
        #     ann_info=dict(seg_map=seg_map)
        #     self.img_dir=os.path.join(self.dataroot,'data',video,'origin/')
        #     self.ann_dir=os.path.join(self.dataroot,'data',video,'mask/')
        #     results = dict(img_info=img_info, ann_info=ann_info)
        #     self.pre_pipeline(results)
        #     clips_img.append(results['img'])
        #     clips_target.append(results['gt_semantic_seg'])
        #     clips_meta.append(results['img_metas'])

        # return dict(clips_img=clips_img,clips_target=clips_target,clips_meta=clips_meta)

        video_imgname=self.img_all[idx]
        video, img_name=video_imgname[0], video_imgname[1]
        img_info=dict(filename=img_name)
        seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        img_info['ann'] = dict(seg_map=seg_map)
        ann_info=dict(seg_map=seg_map)
        img_dir=os.path.join(self.data_root,'data',video,'origin/')
        ann_dir=os.path.join(self.data_root,'data',video,'mask/')
        results = dict(img_info=img_info, ann_info=ann_info)

        # img_info = self.img_infos[idx]
        # ann_info = self.get_ann_info(idx)
        # results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results, img_dir, ann_dir)
        return self.pipeline(results)

    def prepare_train_img2(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        video  = self.videolists[idx]
        imglist = self.imgdic[video]
        imglist_s = imglist[-self.dilation[0]:]
        if len(imglist_s)<1:
            return None
        idx = np.random.choice(list(range(len(imglist_s))))-self.dilation[0]
        this_step=[]
        for dil in self.dilation:
            this_step.append(idx+dil)
        this_step.append(idx)

        clips_img = []
        clips_target=[]
        clips_meta=[]

        print('3',imglist)

        for i in imglist:
        # for i in this_step:
            img_name=imglist[i]
            img_info=dict(filename=img_name)
            seg_map = img_name.replace(img_suffix, seg_map_suffix)
            ann_info=dict(seg_map=seg_map)
            self.img_dir=os.path.join(self.dataroot,'data',video,'origin/')
            self.ann_dir=os.path.join(self.dataroot,'data',video,'mask/')
            results = dict(img_info=img_info, ann_info=ann_info)
            self.pre_pipeline(results)
            clips_img.append(results['img'])
            clips_target.append(results['gt_semantic_seg'])
            clips_meta.append(results['img_metas'])

        return dict(clips_img=clips_img,clips_target=clips_target,clips_meta=clips_meta)


        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """
        video_imgname=self.img_all[idx]
        video, img_name=video_imgname[0], video_imgname[1]
        img_info=dict(filename=img_name)
        seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        img_info['ann'] = dict(seg_map=seg_map)
        img_dir=os.path.join(self.data_root,'data',video,'origin/')
        ann_dir=os.path.join(self.data_root,'data',video,'mask/')

        # img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results, img_dir, ann_dir)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""

        gt_seg_maps = []
        for video_imgname in self.img_all:
            video, img_name=video_imgname[0], video_imgname[1]
            seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
            ann_dir=os.path.join(self.data_root,'data',video,'mask/')

            seg_map = osp.join(ann_dir, seg_map)
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
            gt_seg_maps.append(gt_seg_map)

        # gt_seg_maps = []
        # for img_info in self.img_infos:
        #     seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
        #     if efficient_test:
        #         gt_seg_map = seg_map
        #     else:
        #         gt_seg_map = mmcv.imread(
        #             seg_map, flag='unchanged', backend='pillow')
        #     gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
                         [i]] = summary_table_data[1][i] / 100.0
        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results

@DATASETS.register_module()
class CustomDataset_video2(Dataset):
    """Custom dataset for video semantic segmentation. An example of file structure
    is as followed.

    return video clips instead of sepearate frames

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── video
        │   │   │   ├── origin
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── mask
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 dilation=[-4,-3,-2,-1],
                 clipnum=None):
        # self.pipeline = Compose(pipeline)
        if split=='train':
            self.pipeline_load = Compose(pipeline[:2])
            self.pipeline_process = Compose(pipeline[2:])
        else:
            self.pipeline_load = Compose(pipeline[:1])
            self.pipeline_process = Compose(pipeline[1:])

        self.img_dir = ''
        self.img_suffix = img_suffix
        self.ann_dir = ''
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        self.dilation=dilation
        self.clipnum=clipnum

        # join paths if data_root is specified
        # if self.data_root is not None:
        #     if not osp.isabs(self.img_dir):
        #         self.img_dir = osp.join(self.data_root, self.img_dir)
        #     if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
        #         self.ann_dir = osp.join(self.data_root, self.ann_dir)
        #     if not (self.split is None or osp.isabs(self.split)):
        #         self.split = osp.join(self.data_root, self.split)

        with open(os.path.join(self.data_root,self.split+'.txt')) as f:
            lines=f.readlines()
            self.videolists = [line[:-1] for line in lines]
        
        
        # total_len=0
        self.imgdic={}
        self.img_all=[]
        for video in self.videolists:
            v_path = os.path.join(self.data_root,'data',video,'origin')
            imglist = sorted(os.listdir(v_path))
            # total_len += len(imglist)
            self.imgdic[video]=imglist
            self.img_all=self.img_all+[[video, img] for img in imglist]


        # self.flip_video=False
        self.flip_video=True
        print("flip video: ",self.flip_video)
        # print(self.label_map)
        # print(self.CLASSES, self.PALETTE)
        # exit()

        # if self.split=='train':
            # self.img_all=self.img_all[:500]
        # load annotations
        # self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
        #                                        self.ann_dir,
        #                                        self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        # return len(self.img_infos)
        if self.split=='train':
            return len(self.videolists)
        else:
            return len(self.img_all)
        # return len(self.img_all)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results, img_dir,  ann_dir):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = img_dir
        results['seg_prefix'] = ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img2(idx)
        else:
            return self.prepare_train_img2(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        # video  = self.videolists[idx]
        # imglist = self.imgdic[video]
        # imglist_s = imglist[-self.dilation[0]:]
        # if len(imglist_s)<1:
        #     return None
        # idx = np.random.choice(list(range(len(imglist_s))))-self.dilation[0]
        # this_step=[]
        # for dil in self.dilation:
        #     this_step.append(idx+dil)
        # this_step.append(idx)

        # clips_img = []
        # clips_target=[]
        # clips_meta=[]
        # for i in this_step:
        #     img_name=imglist[i]
        #     img_info=dict(filename=img_name)
        #     seg_map = img_name.replace(img_suffix, seg_map_suffix)
        #     ann_info=dict(seg_map=seg_map)
        #     self.img_dir=os.path.join(self.dataroot,'data',video,'origin/')
        #     self.ann_dir=os.path.join(self.dataroot,'data',video,'mask/')
        #     results = dict(img_info=img_info, ann_info=ann_info)
        #     self.pre_pipeline(results)
        #     clips_img.append(results['img'])
        #     clips_target.append(results['gt_semantic_seg'])
        #     clips_meta.append(results['img_metas'])

        # return dict(clips_img=clips_img,clips_target=clips_target,clips_meta=clips_meta)

        video_imgname=self.img_all[idx]
        video, img_name=video_imgname[0], video_imgname[1]
        img_info=dict(filename=img_name)
        seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        img_info['ann'] = dict(seg_map=seg_map)
        ann_info=dict(seg_map=seg_map)
        img_dir=os.path.join(self.data_root,'data',video,'origin/')
        ann_dir=os.path.join(self.data_root,'data',video,'mask/')
        results = dict(img_info=img_info, ann_info=ann_info)

        # img_info = self.img_infos[idx]
        # ann_info = self.get_ann_info(idx)
        # results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results, img_dir, ann_dir)
        return self.pipeline(results)

    def prepare_train_img2(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        video  = self.videolists[idx]
        imglist = self.imgdic[video]
        ## inverse video with probability of 0.5
        if self.flip_video:
            # print("here")
            if random.random()<0.5:
                imglist=imglist[::-1]
        imglist_s = imglist[-self.dilation[0]:]
        if len(imglist_s)<1:
            return None
        idx = np.random.choice(list(range(len(imglist_s))))-self.dilation[0]
        this_step=[]
        for dil in self.dilation:
            this_step.append(idx+dil)
        this_step.append(idx)

        clips_img = []
        clips_target=[]
        clips_meta=[]
        results_all=[]

        img_info_clips, ann_info_clips, seg_fields_clips, img_prefix_clips, seg_prefix_clips, filename_clips=[],[],[],[],[],[]
        ori_filename_clips, img_clips, img_shape_clips, ori_shape_clips, pad_shape_clips=[],[],[],[],[]
        scale_factor_clips, img_norm_cfg_clips, gt_semantic_seg_clips=[],[],[]

        # print('test',len(imglist[:30:1]))

        # for i in list(range(len(imglist[:30:1]))):
        for i in this_step:
            img_name=imglist[i]
            img_info=dict(filename=img_name)
            seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
            img_info['ann'] = dict(seg_map=seg_map)
            ann_info=dict(seg_map=seg_map)
            img_dir=os.path.join(self.data_root,'data',video,'origin/')
            ann_dir=os.path.join(self.data_root,'data',video,'mask/')
            results = dict(img_info=img_info, ann_info=ann_info)
            self.pre_pipeline(results, img_dir, ann_dir)
            # clips_img.append(results['img'])
            # clips_target.append(results['gt_semantic_seg'])
            # clips_meta.append(results['img_metas'])
            self.pipeline_load(results)
            results_all.append(results)
            img_info_clips.append(results['img_info'])
            ann_info_clips.append(results['ann_info'])
            seg_fields_clips.append(results["seg_fields"])
            img_prefix_clips.append(results["img_prefix"])
            seg_prefix_clips.append(results["seg_prefix"])
            filename_clips.append(results["filename"])
            ori_filename_clips.append(results["ori_filename"])
            img_clips.append(results["img"]) 
            img_shape_clips.append(results["img_shape"])
            ori_shape_clips.append(results["ori_shape"])
            pad_shape_clips.append(results["pad_shape"])
            scale_factor_clips.append(results["scale_factor"])
            img_norm_cfg_clips.append(results["img_norm_cfg"])
            gt_semantic_seg_clips.append(results["gt_semantic_seg"])
            # for key, value in results.item():
            # print(results["seg_fields"])
            # exit()

        results_new=dict(img_info=img_info_clips[-1],ann_info=ann_info_clips[-1],seg_fields=seg_fields_clips[-1],
            img_prefix=img_prefix_clips[-1],seg_prefix=seg_prefix_clips[-1],
            filename=filename_clips[-1],ori_filename=ori_filename_clips[-1],img=img_clips,
            img_shape=img_shape_clips[-1],ori_shape=ori_shape_clips[-1],
            pad_shape=pad_shape_clips[-1],scale_factor=scale_factor_clips[-1],
            img_norm_cfg=img_norm_cfg_clips[-1],gt_semantic_seg=gt_semantic_seg_clips)

        # self.pipeline_process(results_new)

        # print(results_new.keys())

        # exit()

        return self.pipeline_process(results_new)

        # img_info = self.img_infos[idx]
        # ann_info = self.get_ann_info(idx)
        # results = dict(img_info=img_info, ann_info=ann_info)
        # self.pre_pipeline(results)
        # return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """
        video_imgname=self.img_all[idx]
        video, img_name=video_imgname[0], video_imgname[1]
        img_info=dict(filename=img_name)
        seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        img_info['ann'] = dict(seg_map=seg_map)
        img_dir=os.path.join(self.data_root,'data',video,'origin/')
        ann_dir=os.path.join(self.data_root,'data',video,'mask/')

        # img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results, img_dir, ann_dir)
        return self.pipeline(results)

    def prepare_test_img2(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """
        video_imgname=self.img_all[idx]
        video, img_name=video_imgname[0], video_imgname[1]
        imglist = self.imgdic[video]
        img_index=imglist.index(img_name)
        this_step=[]
        for dil in self.dilation:
            clip_index=img_index+dil
            if clip_index>=0 and clip_index<len(imglist):
                this_step.append(clip_index)
        this_step.append(img_index)

        if self.dilation==[-9,-6,-3]:
            if img_index==3:
                this_step=[0,1,2,3]
            elif img_index==4:
                this_step=[0,2,3,4]
            elif img_index==5:
                this_step=[0,2,4,5]
            elif img_index==6:
                this_step=[0,2,4,6]
            elif img_index==7:
                this_step=[0,3,5,7]
            elif img_index==8:
                this_step=[0,3,6,8]

        clips_img = []
        clips_target=[]
        clips_meta=[]
        results_all=[]

        img_info_clips, ann_info_clips, seg_fields_clips, img_prefix_clips, seg_prefix_clips, filename_clips=[],[],[],[],[],[]
        ori_filename_clips, img_clips, img_shape_clips, ori_shape_clips, pad_shape_clips=[],[],[],[],[]
        scale_factor_clips, img_norm_cfg_clips, gt_semantic_seg_clips=[],[],[]

        for i in this_step:
            img_name=imglist[i]
            img_info=dict(filename=img_name)
            seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
            img_info['ann'] = dict(seg_map=seg_map)
            ann_info=dict(seg_map=seg_map)
            img_dir=os.path.join(self.data_root,'data',video,'origin/')
            ann_dir=os.path.join(self.data_root,'data',video,'mask/')
            results = dict(img_info=img_info, ann_info=ann_info)
            self.pre_pipeline(results, img_dir, ann_dir)
            # clips_img.append(results['img'])
            # clips_target.append(results['gt_semantic_seg'])
            # clips_meta.append(results['img_metas'])
            self.pipeline_load(results)
            results_all.append(results)
            img_info_clips.append(results['img_info'])
            ann_info_clips.append(results['ann_info'])
            seg_fields_clips.append(results["seg_fields"])
            img_prefix_clips.append(results["img_prefix"])
            seg_prefix_clips.append(results["seg_prefix"])
            filename_clips.append(results["filename"])
            ori_filename_clips.append(results["ori_filename"])
            img_clips.append(results["img"]) 
            img_shape_clips.append(results["img_shape"])
            ori_shape_clips.append(results["ori_shape"])
            pad_shape_clips.append(results["pad_shape"])
            scale_factor_clips.append(results["scale_factor"])
            img_norm_cfg_clips.append(results["img_norm_cfg"])
            # gt_semantic_seg_clips.append(results["gt_semantic_seg"])
            # for key, value in results.item():
            # print(results["seg_fields"])
            # exit()

        results_new=dict(img_info=img_info_clips[-1],ann_info=ann_info_clips[-1],seg_fields=seg_fields_clips[-1],
            img_prefix=img_prefix_clips[-1],seg_prefix=seg_prefix_clips[-1],
            filename=filename_clips[-1],ori_filename=ori_filename_clips[-1],img=img_clips,
            img_shape=img_shape_clips[-1],ori_shape=ori_shape_clips[-1],
            pad_shape=pad_shape_clips[-1],scale_factor=scale_factor_clips[-1],
            img_norm_cfg=img_norm_cfg_clips[-1])

        # self.pipeline_process(results_new)

        # print(results_new.keys())

        # exit()

        return self.pipeline_process(results_new)

        # img_info=dict(filename=img_name)
        # seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        # img_info['ann'] = dict(seg_map=seg_map)
        # img_dir=os.path.join(self.data_root,'data',video,'origin/')
        # ann_dir=os.path.join(self.data_root,'data',video,'mask/')

        # # img_info = self.img_infos[idx]
        # results = dict(img_info=img_info)
        # self.pre_pipeline(results, img_dir, ann_dir)
        # return self.pipeline(results)



    def format_results(self, results, save_path=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        # pass
        ## changed by guosun
        assert len(results)==len(self.img_all)
        palette_list=[]
        for kk in self.PALETTE:
            palette_list=palette_list+kk

        for ii in range(len(results)):
            result=results[ii]
            video_imgname=self.img_all[ii]
            video, img_name=video_imgname[0], video_imgname[1]
            if isinstance(result, str):
                result = np.load(result)
            save_path_ii=os.path.join(save_path,'result_submission',video,img_name.replace(self.img_suffix, self.seg_map_suffix))
            save_path_directory=os.path.dirname(save_path_ii)
            if not os.path.exists(save_path_directory):
                os.makedirs(save_path_directory)
            res = Image.fromarray(result.astype(np.uint8), mode='P')
            res.putpalette(palette_list)
            res.save(save_path_ii)

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""

        gt_seg_maps = []
        for video_imgname in self.img_all:
            video, img_name=video_imgname[0], video_imgname[1]
            seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
            ann_dir=os.path.join(self.data_root,'data',video,'mask/')

            seg_map = osp.join(ann_dir, seg_map)
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
            gt_seg_maps.append(gt_seg_map)

        # gt_seg_maps = []
        # for img_info in self.img_infos:
        #     seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
        #     if efficient_test:
        #         gt_seg_map = seg_map
        #     else:
        #         gt_seg_map = mmcv.imread(
        #             seg_map, flag='unchanged', backend='pillow')
        #     gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
                         [i]] = summary_table_data[1][i] / 100.0
        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results

@DATASETS.register_module()
class CustomDataset_video3(Dataset):
    """Custom dataset for video semantic segmentation. An example of file structure
    is as followed.

    return video clips instead of sepearate frames

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── video
        │   │   │   ├── origin
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── mask
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = None
    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 dilation=[-4,-3,-2,-1],
                 clipnum=None):
        # self.pipeline = Compose(pipeline)
        if split=='train':
            self.pipeline_load = Compose(pipeline[:2])
            self.pipeline_process = Compose(pipeline[2:])
        else:
            self.pipeline_load = Compose(pipeline[:1])
            self.pipeline_process = Compose(pipeline[1:])

        self.img_dir = ''
        self.img_suffix = img_suffix
        self.ann_dir = ''
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        
        # self.train_class = ['wall', 'ceiling', 'door', 'stair', 'ladder', 'escalator', 'Playground_slide', 'handrail_or_fence', 'window', 'rail', 'goal', 'pillar', 'pole', 'floor', 'ground', 'grass', 'sand', 'athletic_field', 'road', 'path', 'crosswalk', 'building', 'house', 'bridge', 'tower', 'windmill', 'well_or_well_lid', 'other_construction', 'sky', 'mountain', 'stone', 'wood', 'ice', 'snowfield', 'grandstand', 'sea', 'river', 'lake', 'waterfall', 'water', 'billboard_or_Bulletin_Board', 'sculpture', 'pipeline', 'flag', 'parasol_or_umbrella', 'cushion_or_carpet', 'tent', 'roadblock', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'wheeled_machine', 'ship_or_boat', 'raft', 'airplane', 'tyre', 'traffic_light', 'lamp', 'person', 'cat', 'dog', 'horse', 'cattle', 'other_animal', 'tree', 'flower', 'other_plant', 'toy', 'ball_net', 'backboard', 'skateboard', 'bat', 'ball', 'cupboard_or_showcase_or_storage_rack', 'box', 'traveling_case_or_trolley_case', 'basket', 'bag_or_package', 'trash_can', 'cage', 'plate', 'tub_or_bowl_or_pot', 'bottle_or_cup', 'barrel', 'fishbowl', 'bed', 'pillow', 'table_or_desk', 'chair_or_seat', 'bench', 'sofa', 'shelf', 'bathtub', 'gun', 'commode', 'roaster', 'other_machine', 'refrigerator', 'washing_machine', 'Microwave_oven', 'fan', 'curtain', 'textiles', 'clothes', 'painting_or_poster', 'mirror', 'flower_pot_or_vase', 'clock', 'book']
        # self.train_plate = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7], [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255], [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255], [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0], [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255], [0, 214, 255]]
        
        # self.test_class =['tool', 'blackboard', 'tissue', 'screen_or_television', 'computer', 'printer', 'Mobile_phone', 'keyboard', 'other_electronic_product', 'fruit', 'food', 'instrument', 'train']
        # self.test_plate = [[255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163], [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0]]
        
        self.train_class = ['wall', 'ceiling', 'door', 'stair', 'ladder', 'escalator', 'Playground_slide', 'handrail_or_fence', 'window', 'rail', 'goal', 'pillar', 'pole', 'floor', 'ground', 'grass', 'sand', 'athletic_field', 'road', 'path', 'crosswalk', 'building', 'house', 'bridge', 'tower', 'windmill', 'well_or_well_lid', 'other_construction', 'sky', 'mountain', 'stone', 'wood', 'ice', 'snowfield', 'grandstand', 'sea', 'river', 'lake', 'waterfall', 'water', 'billboard_or_Bulletin_Board', 'sculpture', 'pipeline', 'flag', 'parasol_or_umbrella', 'cushion_or_carpet', 'tent', 'roadblock', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'wheeled_machine', 'ship_or_boat', 'raft', 'airplane', 'tyre', 'traffic_light', 'lamp', 'person', 'cat', 'dog', 'horse', 'cattle', 'other_animal', 'tree', 'flower', 'other_plant', 'toy', 'ball_net', 'backboard', 'skateboard', 'bat', 'ball', 'cupboard_or_showcase_or_storage_rack', 'box', 'traveling_case_or_trolley_case', 'basket', 'bag_or_package']
        self.train_plate = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7], [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255], [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255]]
        
        self.test_class = ['trash_can', 'cage', 'plate', 'tub_or_bowl_or_pot', 'bottle_or_cup', 'barrel', 'fishbowl', 'bed', 'pillow', 'table_or_desk', 'chair_or_seat', 'bench', 'sofa', 'shelf', 'bathtub', 'gun', 'commode', 'roaster', 'other_machine', 'refrigerator', 'washing_machine', 'Microwave_oven', 'fan', 'curtain', 'textiles', 'clothes', 'painting_or_poster', 'mirror', 'flower_pot_or_vase', 'clock', 'book', 'tool', 'blackboard', 'tissue', 'screen_or_television', 'computer', 'printer', 'Mobile_phone', 'keyboard', 'other_electronic_product', 'fruit', 'food', 'instrument', 'train']
        self.test_plate = [[255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0], [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163], [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0]]
        
        # self.train_class = ['wall', 'ceiling', 'stair', 'ladder', 'escalator', 'handrail_or_fence', 'window', 'rail', 'pillar', 'pole', 'floor', 'ground', 'grass', 'sand', 'athletic_field', 'road', 'path', 'crosswalk', 'house', 'bridge', 'tower', 'windmill', 'well_or_well_lid', 'other_construction', 'sky', 'mountain', 'stone', 'wood', 'ice', 'snowfield', 'grandstand', 'sea', 'river', 'waterfall', 'water', 'billboard_or_Bulletin_Board', 'sculpture', 'pipeline', 'parasol_or_umbrella', 'cushion_or_carpet', 'tent', 'roadblock', 'car', 'bus', 'truck', 'bicycle', 'wheeled_machine', 'ship_or_boat', 'raft', 'airplane', 'tyre', 'traffic_light', 'person', 'cat', 'dog', 'horse', 'cattle', 'other_animal', 'tree', 'flower', 'other_plant', 'toy', 'ball_net', 'backboard', 'skateboard', 'bat', 'ball', 'cupboard_or_showcase_or_storage_rack', 'box', 'traveling_case_or_trolley_case', 'bag_or_package', 'trash_can', 'cage', 'plate', 'tub_or_bowl_or_pot', 'bottle_or_cup', 'fishbowl', 'bed', 'pillow', 'table_or_desk', 'chair_or_seat', 'bench', 'sofa', 'shelf', 'bathtub', 'gun', 'commode', 'roaster', 'refrigerator', 'washing_machine', 'Microwave_oven', 'curtain', 'textiles', 'clothes', 'painting_or_poster', 'mirror', 'flower_pot_or_vase', 'clock', 'book', 'tool', 'blackboard', 'tissue', 'screen_or_television', 'printer', 'Mobile_phone', 'keyboard', 'other_electronic_product', 'fruit', 'food', 'instrument', 'train']
        # self.train_plate = [[120, 120, 120], [180, 120, 120], [80, 50, 50], [4, 200, 3], [120, 120, 80], [204, 5, 255], [230, 230, 230], [4, 250, 7], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200], [255, 6, 51], [11, 102, 255], [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8], [255, 61, 6], [255, 194, 7], [255, 122, 8], [0, 255, 20], [255, 8, 41], [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 0, 255], [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255], [0, 255, 163], [255, 153, 0], [255, 112, 0], [143, 255, 0], [82, 0, 255], [255, 235, 0], [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163], [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0]]
        
        # self.test_class = ['door', 'Playground_slide', 'goal', 'building', 'lake', 'flag', 'motorcycle', 'lamp', 'basket', 'barrel', 'other_machine', 'fan', 'computer']
        # self.test_plate = [[6, 230, 230], [140, 140, 140], [224, 5, 255], [61, 230, 250], [102, 8, 255], [255, 5, 153], [255, 31, 0], [31, 0, 255], [255, 92, 0], [0, 31, 255], [0, 255, 10], [163, 255, 0], [70, 184, 160]]
        
        self.test_setting = 'val_all'
        # remember to modify setting in cat_zeroshot_classifier.py line 34-35
        
        # self.custom_classes = False
        
        if split=='train':
            self.mask_file = 'train_seen'
            self.CLASSES, self.PALETTE = self.get_classes_and_palette(self.train_class, self.train_plate)
            # self.CLASSES, self.PALETTE = self.train_class,self.train_plate
        else:
            # print(self.split)
            # exit()
            if self.test_setting == 'val_all':
                self.mask_file = 'val_all'
                # self.CLASSES,self.PALETTE = self.train_class+self.test_class, self.train_plate+self.test_plate
                self.CLASSES, self.PALETTE = self.get_classes_and_palette(
                self.train_class+self.test_class,  self.train_plate+self.test_plate)
            elif self.test_setting == 'val_seen':
                self.mask_file = 'val_seen'
                # self.CLASSES,self.PALETTE = self.train_class, self.train_plate
                self.CLASSES, self.PALETTE = self.get_classes_and_palette(
                self.train_class, self.train_plate)
                
            elif self.test_setting == 'val_unseen':
                self.mask_file = 'val_unseen'
                # self.CLASSES,self.PALETTE = self.test_class, self.test_plate
                self.CLASSES, self.PALETTE = self.get_classes_and_palette(
                self.test_class, self.test_plate)
        

        self.dilation=dilation
        self.clipnum=clipnum

        # join paths if data_root is specified
        # if self.data_root is not None:
        #     if not osp.isabs(self.img_dir):
        #         self.img_dir = osp.join(self.data_root, self.img_dir)
        #     if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
        #         self.ann_dir = osp.join(self.data_root, self.ann_dir)
        #     if not (self.split is None or osp.isabs(self.split)):
        #         self.split = osp.join(self.data_root, self.split)
        

        with open(os.path.join(self.data_root,self.split+'.txt')) as f:
            lines=f.readlines()
            self.videolists = [line[:-1] for line in lines]
        
        # total_len = 0
        
        self.imgdic={}
        self.img_all=[]
        for video in self.videolists:
            v_path = os.path.join(self.data_root,'data',video,'origin')
            imglist = sorted(os.listdir(v_path))
            # total_len += len(imglist)
            self.imgdic[video]=imglist
            self.img_all=self.img_all+[[video, img] for img in imglist]

        # print(total_len)
        # exit()

        # self.flip_video=False
        self.flip_video=True
        print("flip video: ",self.flip_video)
        # print(self.label_map)
        # exit()
        # print(self.CLASSES,len(self.CLASSES))
        # exit()

        # if self.split=='train':
            # self.img_all=self.img_all[:500]
        # load annotations
        # self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
        #                                        self.ann_dir,
        #                                        self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        # return len(self.img_infos)
        
        if self.split=='train':
            return len(self.videolists)
        else:
            return len(self.img_all)
        # return len(self.img_all)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results, img_dir,  ann_dir):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = img_dir
        results['seg_prefix'] = ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img2(idx)
        else:
            return self.prepare_train_img2(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        # video  = self.videolists[idx]
        # imglist = self.imgdic[video]
        # imglist_s = imglist[-self.dilation[0]:]
        # if len(imglist_s)<1:
        #     return None
        # idx = np.random.choice(list(range(len(imglist_s))))-self.dilation[0]
        # this_step=[]
        # for dil in self.dilation:
        #     this_step.append(idx+dil)
        # this_step.append(idx)

        # clips_img = []
        # clips_target=[]
        # clips_meta=[]
        # for i in this_step:
        #     img_name=imglist[i]
        #     img_info=dict(filename=img_name)
        #     seg_map = img_name.replace(img_suffix, seg_map_suffix)
        #     ann_info=dict(seg_map=seg_map)
        #     self.img_dir=os.path.join(self.dataroot,'data',video,'origin/')
        #     self.ann_dir=os.path.join(self.dataroot,'data',video,'mask/')
        #     results = dict(img_info=img_info, ann_info=ann_info)
        #     self.pre_pipeline(results)
        #     clips_img.append(results['img'])
        #     clips_target.append(results['gt_semantic_seg'])
        #     clips_meta.append(results['img_metas'])

        # return dict(clips_img=clips_img,clips_target=clips_target,clips_meta=clips_meta)

        video_imgname=self.img_all[idx]
        video, img_name=video_imgname[0], video_imgname[1]
        img_info=dict(filename=img_name)
        seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        img_info['ann'] = dict(seg_map=seg_map)
        ann_info=dict(seg_map=seg_map)
        img_dir=os.path.join(self.data_root,'data',video,'origin/')
        # ann_dir=os.path.join(self.data_root,'data',video,'mask/')
        ann_dir=os.path.join(self.data_root,'Detectron',self.mask_file,'data',video)
        results = dict(img_info=img_info, ann_info=ann_info)

        # img_info = self.img_infos[idx]
        # ann_info = self.get_ann_info(idx)
        # results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results, img_dir, ann_dir)
        return self.pipeline(results)

    def prepare_train_img2(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        video  = self.videolists[idx]
        imglist = self.imgdic[video]
        ## inverse video with probability of 0.5
        if self.flip_video:
            # print("here")
            if random.random()<0.5:
                imglist=imglist[::-1]
        imglist_s = imglist[-self.dilation[0]:]
        if len(imglist_s)<1:
            return None
        idx = np.random.choice(list(range(len(imglist_s))))-self.dilation[0]
        this_step=[]
        for dil in self.dilation:
            this_step.append(idx+dil)
        this_step.append(idx)

        clips_img = []
        clips_target=[]
        clips_meta=[]
        results_all=[]

        img_info_clips, ann_info_clips, seg_fields_clips, img_prefix_clips, seg_prefix_clips, filename_clips=[],[],[],[],[],[]
        ori_filename_clips, img_clips, img_shape_clips, ori_shape_clips, pad_shape_clips=[],[],[],[],[]
        scale_factor_clips, img_norm_cfg_clips, gt_semantic_seg_clips=[],[],[]

        # print('test',len(imglist[:30:1]))
        # print('ttt ',len(self.videolists))
        # print('see',len(imglist))

        # for i in list(range(len(imglist[:30:1]))):
        # for i in [this_step[-1]]:
        for i in this_step:
            img_name=imglist[i]
            img_info=dict(filename=img_name)
            seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
            img_info['ann'] = dict(seg_map=seg_map)
            ann_info=dict(seg_map=seg_map)
            img_dir=os.path.join(self.data_root,'data',video,'origin/')
            # ann_dir=os.path.join(self.data_root,'data',video,'mask/')
            ann_dir=os.path.join(self.data_root,'Detectron',self.mask_file,'data',video)
            results = dict(img_info=img_info, ann_info=ann_info)
            self.pre_pipeline(results, img_dir, ann_dir)
            # clips_img.append(results['img'])
            # clips_target.append(results['gt_semantic_seg'])
            # clips_meta.append(results['img_metas'])
            self.pipeline_load(results)
            results_all.append(results)
            img_info_clips.append(results['img_info'])
            ann_info_clips.append(results['ann_info'])
            seg_fields_clips.append(results["seg_fields"])
            img_prefix_clips.append(results["img_prefix"])
            seg_prefix_clips.append(results["seg_prefix"])
            filename_clips.append(results["filename"])
            ori_filename_clips.append(results["ori_filename"])
            img_clips.append(results["img"]) 
            img_shape_clips.append(results["img_shape"])
            ori_shape_clips.append(results["ori_shape"])
            pad_shape_clips.append(results["pad_shape"])
            scale_factor_clips.append(results["scale_factor"])
            img_norm_cfg_clips.append(results["img_norm_cfg"])
            gt_semantic_seg_clips.append(results["gt_semantic_seg"])
            
            # print("shape",len(gt_semantic_seg_clips),gt_semantic_seg_clips[0].shape)
            # exit()
            
            # sem_seg_gt = torch.as_tensor(results['gt_semantic_seg'].astype("long"))
            # sem_seg = sem_seg_gt.numpy()
            # classes = np.unique(sem_seg)
            # classes = classes[classes != self.ignore_index]
            
            # gt_classes = torch.tensor(classes,dtype=torch.int64)
            # masks = []
            # for class_id in classes:
            #     masks.append(sem_seg == class_id)
            # if len(masks) == 0:
            #     gt_mask = torch.zeros((0, sem_seg.shape[-2], sem_seg.shape[-1]))
            # else:
            #     gt_mask = [[torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])]]
            
            # print(gt_mask[0].shape)
            # exit()

        # results_new=dict(img_info=img_info_clips[-1],ann_info=ann_info_clips[-1],seg_fields=seg_fields_clips[-1],
        #     img_prefix=img_prefix_clips[-1],seg_prefix=seg_prefix_clips[-1],
        #     filename=filename_clips[-1],ori_filename=ori_filename_clips[-1],img=img_clips,
        #     img_shape=img_shape_clips[-1],ori_shape=ori_shape_clips[-1],
        #     pad_shape=pad_shape_clips[-1],scale_factor=scale_factor_clips[-1],
        #     img_norm_cfg=img_norm_cfg_clips[-1],gt_semantic_seg=gt_semantic_seg_clips)
        
        results_new=dict(img_info=img_info_clips[-1],ann_info=ann_info_clips[-1],seg_fields=seg_fields_clips[-1],
            img_prefix=img_prefix_clips[-1],seg_prefix=seg_prefix_clips[-1],
            filename=filename_clips[-1],ori_filename=ori_filename_clips[-1],img=img_clips,
            img_shape=img_shape_clips[-1],ori_shape=ori_shape_clips[-1],
            pad_shape=pad_shape_clips[-1],scale_factor=scale_factor_clips[-1],
            img_norm_cfg=img_norm_cfg_clips[-1],gt_semantic_seg=gt_semantic_seg_clips)

        # self.pipeline_process(results_new)

        # print(results_new.keys())

        # exit()

        return self.pipeline_process(results_new)

        # img_info = self.img_infos[idx]
        # ann_info = self.get_ann_info(idx)
        # results = dict(img_info=img_info, ann_info=ann_info)
        # self.pre_pipeline(results)
        # return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """
        video_imgname=self.img_all[idx]
        video, img_name=video_imgname[0], video_imgname[1]
        img_info=dict(filename=img_name)
        seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        img_info['ann'] = dict(seg_map=seg_map)
        img_dir=os.path.join(self.data_root,'data',video,'origin/')
        # ann_dir=os.path.join(self.data_root,'data',video,'mask/')
        ann_dir=os.path.join(self.data_root,'Detectron',self.mask_file,'data',video)

        # img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results, img_dir, ann_dir)
        return self.pipeline(results)

    def prepare_test_img2(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """
        video_imgname=self.img_all[idx]
        video, img_name=video_imgname[0], video_imgname[1]
        imglist = self.imgdic[video]
        img_index=imglist.index(img_name)
        
        this_step=[]
        this_steps=[]
        for dil in self.dilation:
            clip_index=img_index+dil
            if clip_index>=0 and clip_index<len(imglist):
                this_step.append(clip_index)
        this_step.append(img_index)
        
        if self.dilation==[-9,-6,-3]:
            if img_index==3:
                this_step=[0,1,2,3]
            elif img_index==4:
                this_step=[0,2,3,4]
            elif img_index==5:
                this_step=[0,2,4,5]
            elif img_index==6:
                this_step=[0,2,4,6]
            elif img_index==7:
                this_step=[0,3,5,7]
            elif img_index==8:
                this_step=[0,3,6,8]
                
        # print(idx,video_imgname,video,img_name,len(imglist),img_index,this_step)
        

        clips_img = []
        clips_target=[]
        clips_meta=[]
        results_all=[]

        img_info_clips, ann_info_clips, seg_fields_clips, img_prefix_clips, seg_prefix_clips, filename_clips=[],[],[],[],[],[]
        ori_filename_clips, img_clips, img_shape_clips, ori_shape_clips, pad_shape_clips=[],[],[],[],[]
        scale_factor_clips, img_norm_cfg_clips, gt_semantic_seg_clips=[],[],[]

        # for i in list(range(len(imglist[:15:1]))):
        
        for i in this_step:
            img_name=imglist[i]
            img_info=dict(filename=img_name)
            seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
            img_info['ann'] = dict(seg_map=seg_map)
            ann_info=dict(seg_map=seg_map)
            img_dir=os.path.join(self.data_root,'data',video,'origin/')
            # ann_dir=os.path.join(self.data_root,'data',video,'mask/')
            ann_dir=os.path.join(self.data_root,'Detectron',self.mask_file,'data',video)
            results = dict(img_info=img_info, ann_info=ann_info)
            self.pre_pipeline(results, img_dir, ann_dir)
            # clips_img.append(results['img'])
            # clips_target.append(results['gt_semantic_seg'])
            # clips_meta.append(results['img_metas'])
            self.pipeline_load(results)
            results_all.append(results)
            img_info_clips.append(results['img_info'])
            ann_info_clips.append(results['ann_info'])
            seg_fields_clips.append(results["seg_fields"])
            img_prefix_clips.append(results["img_prefix"])
            seg_prefix_clips.append(results["seg_prefix"])
            filename_clips.append(results["filename"])
            ori_filename_clips.append(results["ori_filename"])
            img_clips.append(results["img"]) 
            img_shape_clips.append(results["img_shape"])
            ori_shape_clips.append(results["ori_shape"])
            pad_shape_clips.append(results["pad_shape"])
            scale_factor_clips.append(results["scale_factor"])
            img_norm_cfg_clips.append(results["img_norm_cfg"])
        
        # if this_step[-1]<6:
        #     for i in this_step:
        #         img_name=imglist[i]
        #         img_info=dict(filename=img_name)
        #         seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        #         img_info['ann'] = dict(seg_map=seg_map)
        #         ann_info=dict(seg_map=seg_map)
        #         img_dir=os.path.join(self.data_root,'data',video,'origin/')
        #         ann_dir=os.path.join(self.data_root,'data',video,'mask/')
        #         results = dict(img_info=img_info, ann_info=ann_info)
        #         self.pre_pipeline(results, img_dir, ann_dir)
        #         # clips_img.append(results['img'])
        #         # clips_target.append(results['gt_semantic_seg'])
        #         # clips_meta.append(results['img_metas'])
        #         self.pipeline_load(results)
        #         results_all.append(results)
        #         img_info_clips.append(results['img_info'])
        #         ann_info_clips.append(results['ann_info'])
        #         seg_fields_clips.append(results["seg_fields"])
        #         img_prefix_clips.append(results["img_prefix"])
        #         seg_prefix_clips.append(results["seg_prefix"])
        #         filename_clips.append(results["filename"])
        #         ori_filename_clips.append(results["ori_filename"])
        #         img_clips.append(results["img"]) 
        #         img_shape_clips.append(results["img_shape"])
        #         ori_shape_clips.append(results["ori_shape"])
        #         pad_shape_clips.append(results["pad_shape"])
        #         scale_factor_clips.append(results["scale_factor"])
        #         img_norm_cfg_clips.append(results["img_norm_cfg"])
        # else:
        #     # print(max(0,this_step[-1]-15),this_step[-1])
        #     for i in range(max(0,this_step[-1]-13),this_step[-1]):
        #         # print('i am in',idx,this_step,i)
        #         img_name=imglist[i]
        #         img_info=dict(filename=img_name)
        #         seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        #         img_info['ann'] = dict(seg_map=seg_map)
        #         ann_info=dict(seg_map=seg_map)
        #         img_dir=os.path.join(self.data_root,'data',video,'origin/')
        #         ann_dir=os.path.join(self.data_root,'data',video,'mask/')
        #         results = dict(img_info=img_info, ann_info=ann_info)
        #         self.pre_pipeline(results, img_dir, ann_dir)
        #         # clips_img.append(results['img'])
        #         # clips_target.append(results['gt_semantic_seg'])
        #         # clips_meta.append(results['img_metas'])
        #         self.pipeline_load(results)
        #         results_all.append(results)
        #         img_info_clips.append(results['img_info'])
        #         ann_info_clips.append(results['ann_info'])
        #         seg_fields_clips.append(results["seg_fields"])
        #         img_prefix_clips.append(results["img_prefix"])
        #         seg_prefix_clips.append(results["seg_prefix"])
        #         filename_clips.append(results["filename"])
        #         ori_filename_clips.append(results["ori_filename"])
        #         img_clips.append(results["img"]) 
        #         img_shape_clips.append(results["img_shape"])
        #         ori_shape_clips.append(results["ori_shape"])
        #         pad_shape_clips.append(results["pad_shape"])
        #         scale_factor_clips.append(results["scale_factor"])
        #         img_norm_cfg_clips.append(results["img_norm_cfg"])
                # gt_semantic_seg_clips.append(results["gt_semantic_seg"])
                # for key, value in results.item():
                # print(results["seg_fields"])
                # exit()

        results_new=dict(img_info=img_info_clips[-1],ann_info=ann_info_clips[-1],seg_fields=seg_fields_clips[-1],
            img_prefix=img_prefix_clips[-1],seg_prefix=seg_prefix_clips[-1],
            filename=filename_clips[-1],ori_filename=ori_filename_clips[-1],img=img_clips,
            img_shape=img_shape_clips[-1],ori_shape=ori_shape_clips[-1],
            pad_shape=pad_shape_clips[-1],scale_factor=scale_factor_clips[-1],
            img_norm_cfg=img_norm_cfg_clips[-1])

        # self.pipeline_process(results_new)

        # print(results_new.keys())

        # exit()

        return self.pipeline_process(results_new)

        # img_info=dict(filename=img_name)
        # seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        # img_info['ann'] = dict(seg_map=seg_map)
        # img_dir=os.path.join(self.data_root,'data',video,'origin/')
        # ann_dir=os.path.join(self.data_root,'data',video,'mask/')

        # # img_info = self.img_infos[idx]
        # results = dict(img_info=img_info)
        # self.pre_pipeline(results, img_dir, ann_dir)
        # return self.pipeline(results)



    def format_results(self, results, save_path=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        # pass
        ## changed by guosun
        assert len(results)==len(self.img_all)
        palette_list=[]
        for kk in self.PALETTE:
            palette_list=palette_list+kk

        for ii in range(len(results)):
            result=results[ii]
            video_imgname=self.img_all[ii]
            video, img_name=video_imgname[0], video_imgname[1]
            if isinstance(result, str):
                result = np.load(result)
            save_path_ii=os.path.join(save_path,'result_submission',video,img_name.replace(self.img_suffix, self.seg_map_suffix))
            save_path_directory=os.path.dirname(save_path_ii)
            if not os.path.exists(save_path_directory):
                os.makedirs(save_path_directory)
            res = Image.fromarray(result.astype(np.uint8), mode='P')
            res.putpalette(palette_list)
            res.save(save_path_ii)

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""

        gt_seg_maps = []
        for video_imgname in self.img_all:
            video, img_name=video_imgname[0], video_imgname[1]
            seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
            # ann_dir=os.path.join(self.data_root,'data',video,'mask/')
            ann_dir=os.path.join(self.data_root,'Detectron',self.mask_file,'data',video)
            

            seg_map = osp.join(ann_dir, seg_map)
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
            gt_seg_maps.append(gt_seg_map)
            # print(seg_map,efficient_test,np.unique(gt_seg_map))

        # gt_seg_maps = []
        # for img_info in self.img_infos:
        #     seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
        #     if efficient_test:
        #         gt_seg_map = seg_map
        #     else:
        #         gt_seg_map = mmcv.imread(
        #             seg_map, flag='unchanged', backend='pillow')
        #     gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        # print(self.CLASSES)
        # exit()
        
        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
                         [i]] = summary_table_data[1][i] / 100.0
        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results

@DATASETS.register_module()
class CustomDataset_video4(Dataset):
    """Custom dataset for video semantic segmentation. An example of file structure
    is as followed.

    return video clips instead of sepearate frames

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── video
        │   │   │   ├── origin
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── mask
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = None
    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 dilation=[-4,-3,-2,-1],
                 clipnum=None):
        # self.pipeline = Compose(pipeline)
        if split=='train':
            self.pipeline_load = Compose(pipeline[:2])
            self.pipeline_process = Compose(pipeline[2:])
        else:
            self.pipeline_load = Compose(pipeline[:1])
            self.pipeline_process = Compose(pipeline[1:])

        self.img_dir = ''
        self.img_suffix = img_suffix
        self.ann_dir = ''
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        
        self.train_class = ['wall', 'ceiling', 'door', 'stair', 'ladder', 'escalator', 'Playground_slide', 'handrail_or_fence', 'window', 'rail', 'goal', 'pillar', 'pole', 'floor', 'ground', 'grass', 'sand', 'athletic_field', 'road', 'path', 'crosswalk', 'building', 'house', 'bridge', 'tower', 'windmill', 'well_or_well_lid', 'other_construction', 'sky', 'mountain', 'stone', 'wood', 'ice', 'snowfield', 'grandstand', 'sea', 'river', 'lake', 'waterfall', 'water', 'billboard_or_Bulletin_Board', 'sculpture', 'pipeline', 'flag', 'parasol_or_umbrella', 'cushion_or_carpet', 'tent', 'roadblock', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'wheeled_machine', 'ship_or_boat', 'raft', 'airplane', 'tyre', 'traffic_light', 'lamp', 'person', 'cat', 'dog', 'horse', 'cattle', 'other_animal', 'tree', 'flower', 'other_plant', 'toy', 'ball_net', 'backboard', 'skateboard', 'bat', 'ball', 'cupboard_or_showcase_or_storage_rack', 'box', 'traveling_case_or_trolley_case', 'basket', 'bag_or_package']
        self.train_plate = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7], [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255], [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255]]
        
        self.test_class = ['trash_can', 'cage', 'plate', 'tub_or_bowl_or_pot', 'bottle_or_cup', 'barrel', 'fishbowl', 'bed', 'pillow', 'table_or_desk', 'chair_or_seat', 'bench', 'sofa', 'shelf', 'bathtub', 'gun', 'commode', 'roaster', 'other_machine', 'refrigerator', 'washing_machine', 'Microwave_oven', 'fan', 'curtain', 'textiles', 'clothes', 'painting_or_poster', 'mirror', 'flower_pot_or_vase', 'clock', 'book', 'tool', 'blackboard', 'tissue', 'screen_or_television', 'computer', 'printer', 'Mobile_phone', 'keyboard', 'other_electronic_product', 'fruit', 'food', 'instrument', 'train']
        self.test_plate = [[255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0], [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163], [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0]]
        
        self.test_setting = 'val_unseen'
        # remember to modify setting in cat_zeroshot_classifier.py line 34-35
        
        # self.custom_classes = False
        
        if split=='train':
            self.mask_file = 'train_seen'
            self.CLASSES, self.PALETTE = self.get_classes_and_palette(self.train_class, self.train_plate)
            # self.CLASSES, self.PALETTE = self.train_class,self.train_plate
        else:
            # print(self.split)
            # exit()
            if self.test_setting == 'val_all':
                self.mask_file = 'val_all'
                # self.CLASSES,self.PALETTE = self.train_class+self.test_class, self.train_plate+self.test_plate
                self.CLASSES, self.PALETTE = self.get_classes_and_palette(
                self.train_class+self.test_class,  self.train_plate+self.test_plate)
            elif self.test_setting == 'val_seen':
                self.mask_file = 'val_seen'
                # self.CLASSES,self.PALETTE = self.train_class, self.train_plate
                self.CLASSES, self.PALETTE = self.get_classes_and_palette(
                self.train_class, self.train_plate)
                
            elif self.test_setting == 'val_unseen':
                self.mask_file = 'val_unseen'
                # self.CLASSES,self.PALETTE = self.test_class, self.test_plate
                self.CLASSES, self.PALETTE = self.get_classes_and_palette(
                self.test_class, self.test_plate)
        
        # print(self.train_class,self.test_class)
        # print(self.train_plate+self.test_plate)
        # print(len(self.train_plate+self.test_plate),len(self.train_class+self.test_class))

        self.dilation=dilation
        self.clipnum=clipnum

        # join paths if data_root is specified
        # if self.data_root is not None:
        #     if not osp.isabs(self.img_dir):
        #         self.img_dir = osp.join(self.data_root, self.img_dir)
        #     if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
        #         self.ann_dir = osp.join(self.data_root, self.ann_dir)
        #     if not (self.split is None or osp.isabs(self.split)):
        #         self.split = osp.join(self.data_root, self.split)
        

        with open(os.path.join(self.data_root,self.split+'.txt')) as f:
            lines=f.readlines()
            self.videolists = [line[:-1] for line in lines]
        
        # total_len = 0
        
        self.imgdic={}
        self.img_all=[]
        for video in self.videolists:
            v_path = os.path.join(self.data_root,'data',video,'origin')
            imglist = sorted(os.listdir(v_path))
            # total_len += len(imglist)
            self.imgdic[video]=imglist
            self.img_all=self.img_all+[[video, img] for img in imglist]

        # print(total_len)
        # exit()

        # self.flip_video=False
        self.flip_video=True
        print("flip video: ",self.flip_video)
        # print(self.label_map)
        # exit()
        # print(self.CLASSES,len(self.CLASSES))
        # exit()

        # if self.split=='train':
            # self.img_all=self.img_all[:500]
        # load annotations
        # self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
        #                                        self.ann_dir,
        #                                        self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        # return len(self.img_infos)
        
        if self.split=='train':
            return len(self.videolists)
        else:
            return len(self.img_all)
        # return len(self.img_all)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    # print(img_info)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results, img_dir,  ann_dir):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = img_dir
        results['seg_prefix'] = ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img2(idx)
        else:
            return self.prepare_train_img2(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        # video  = self.videolists[idx]
        # imglist = self.imgdic[video]
        # imglist_s = imglist[-self.dilation[0]:]
        # if len(imglist_s)<1:
        #     return None
        # idx = np.random.choice(list(range(len(imglist_s))))-self.dilation[0]
        # this_step=[]
        # for dil in self.dilation:
        #     this_step.append(idx+dil)
        # this_step.append(idx)

        # clips_img = []
        # clips_target=[]
        # clips_meta=[]
        # for i in this_step:
        #     img_name=imglist[i]
        #     img_info=dict(filename=img_name)
        #     seg_map = img_name.replace(img_suffix, seg_map_suffix)
        #     ann_info=dict(seg_map=seg_map)
        #     self.img_dir=os.path.join(self.dataroot,'data',video,'origin/')
        #     self.ann_dir=os.path.join(self.dataroot,'data',video,'mask/')
        #     results = dict(img_info=img_info, ann_info=ann_info)
        #     self.pre_pipeline(results)
        #     clips_img.append(results['img'])
        #     clips_target.append(results['gt_semantic_seg'])
        #     clips_meta.append(results['img_metas'])

        # return dict(clips_img=clips_img,clips_target=clips_target,clips_meta=clips_meta)

        video_imgname=self.img_all[idx]
        video, img_name=video_imgname[0], video_imgname[1]
        img_info=dict(filename=img_name)
        seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        img_info['ann'] = dict(seg_map=seg_map)
        ann_info=dict(seg_map=seg_map)
        img_dir=os.path.join(self.data_root,'data',video,'origin/')
        # ann_dir=os.path.join(self.data_root,'data',video,'mask/')
        ann_dir=os.path.join(self.data_root,'Detectron',self.mask_file,'data',video)
        results = dict(img_info=img_info, ann_info=ann_info)

        # img_info = self.img_infos[idx]
        # ann_info = self.get_ann_info(idx)
        # results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results, img_dir, ann_dir)
        return self.pipeline(results)

    def prepare_train_img2(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        video  = self.videolists[idx]
        imglist = self.imgdic[video]
        ## inverse video with probability of 0.5
        if self.flip_video:
            # print("here")
            if random.random()<0.5:
                imglist=imglist[::-1]
        imglist_s = imglist[-self.dilation[0]:]
        if len(imglist_s)<1:
            return None
        idx = np.random.choice(list(range(len(imglist_s))))-self.dilation[0]
        this_step=[]
        for dil in self.dilation:
            this_step.append(idx+dil)
        this_step.append(idx)

        clips_img = []
        clips_target=[]
        clips_meta=[]
        results_all=[]

        img_info_clips, ann_info_clips, seg_fields_clips, img_prefix_clips, seg_prefix_clips, filename_clips=[],[],[],[],[],[]
        ori_filename_clips, img_clips, img_shape_clips, ori_shape_clips, pad_shape_clips=[],[],[],[],[]
        scale_factor_clips, img_norm_cfg_clips, gt_semantic_seg_clips=[],[],[]

        # print('test',len(imglist[:30:1]))
        # print('ttt ',len(self.videolists))
        # print('see',len(imglist))

        ran_val = np.random.choice(list(range(len(imglist_s))))
        while ran_val in this_step:
            ran_val = np.random.choice(list(range(len(imglist_s))))
        
        #load random frame
        img_name=imglist[ran_val]
        img_info=dict(filename=img_name)
        seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        img_info['ann'] = dict(seg_map=seg_map)
        ann_info=dict(seg_map=seg_map)
        img_dir=os.path.join(self.data_root,'data',video,'origin/')
        # ann_dir=os.path.join(self.data_root,'data',video,'mask/')
        ann_dir=os.path.join(self.data_root,'Detectron',self.mask_file,'data',video)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results, img_dir, ann_dir)
        self.pipeline_load(results)
        img_clips.append(results["img"]) 
        
        # print("in")
        
        # for i in list(range(len(imglist[:30:1]))):
        # for i in [this_step[-1]]:
        for i in this_step:
            img_name=imglist[i]
            img_info=dict(filename=img_name)
            seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
            img_info['ann'] = dict(seg_map=seg_map)
            ann_info=dict(seg_map=seg_map)
            img_dir=os.path.join(self.data_root,'data',video,'origin/')
            # ann_dir=os.path.join(self.data_root,'data',video,'mask/')
            ann_dir=os.path.join(self.data_root,'Detectron',self.mask_file,'data',video)
            results = dict(img_info=img_info, ann_info=ann_info)
            self.pre_pipeline(results, img_dir, ann_dir)
            # clips_img.append(results['img'])
            # clips_target.append(results['gt_semantic_seg'])
            # clips_meta.append(results['img_metas'])
            self.pipeline_load(results)
            results_all.append(results)
            img_info_clips.append(results['img_info'])
            ann_info_clips.append(results['ann_info'])
            seg_fields_clips.append(results["seg_fields"])
            img_prefix_clips.append(results["img_prefix"])
            seg_prefix_clips.append(results["seg_prefix"])
            filename_clips.append(results["filename"])
            ori_filename_clips.append(results["ori_filename"])
            img_clips.append(results["img"]) 
            img_shape_clips.append(results["img_shape"])
            ori_shape_clips.append(results["ori_shape"])
            pad_shape_clips.append(results["pad_shape"])
            scale_factor_clips.append(results["scale_factor"])
            img_norm_cfg_clips.append(results["img_norm_cfg"])
            if i == this_step[-1]:
                gt_semantic_seg_clips.append(results["gt_semantic_seg"])
        
        
            # print("shape",len(gt_semantic_seg_clips),gt_semantic_seg_clips[0].shape)
            # exit()
            
            # sem_seg_gt = torch.as_tensor(results['gt_semantic_seg'].astype("long"))
            # sem_seg = sem_seg_gt.numpy()
            # classes = np.unique(sem_seg)
            # classes = classes[classes != self.ignore_index]
            
            # gt_classes = torch.tensor(classes,dtype=torch.int64)
            # masks = []
            # for class_id in classes:
            #     masks.append(sem_seg == class_id)
            # if len(masks) == 0:
            #     gt_mask = torch.zeros((0, sem_seg.shape[-2], sem_seg.shape[-1]))
            # else:
            #     gt_mask = [[torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])]]
            
            # print(gt_mask[0].shape)
            # exit()

        # results_new=dict(img_info=img_info_clips[-1],ann_info=ann_info_clips[-1],seg_fields=seg_fields_clips[-1],
        #     img_prefix=img_prefix_clips[-1],seg_prefix=seg_prefix_clips[-1],
        #     filename=filename_clips[-1],ori_filename=ori_filename_clips[-1],img=img_clips,
        #     img_shape=img_shape_clips[-1],ori_shape=ori_shape_clips[-1],
        #     pad_shape=pad_shape_clips[-1],scale_factor=scale_factor_clips[-1],
        #     img_norm_cfg=img_norm_cfg_clips[-1],gt_semantic_seg=gt_semantic_seg_clips)
        
        results_new=dict(img_info=img_info_clips[-1],ann_info=ann_info_clips[-1],seg_fields=seg_fields_clips[-1],
            img_prefix=img_prefix_clips[-1],seg_prefix=seg_prefix_clips[-1],
            filename=filename_clips[-1],ori_filename=ori_filename_clips[-1],img=img_clips,
            img_shape=img_shape_clips[-1],ori_shape=ori_shape_clips[-1],
            pad_shape=pad_shape_clips[-1],scale_factor=scale_factor_clips[-1],
            img_norm_cfg=img_norm_cfg_clips[-1],gt_semantic_seg=gt_semantic_seg_clips)

        # self.pipeline_process(results_new)

        # print(results_new.keys())

        # exit()

        return self.pipeline_process(results_new)

        # img_info = self.img_infos[idx]
        # ann_info = self.get_ann_info(idx)
        # results = dict(img_info=img_info, ann_info=ann_info)
        # self.pre_pipeline(results)
        # return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """
        video_imgname=self.img_all[idx]
        video, img_name=video_imgname[0], video_imgname[1]
        img_info=dict(filename=img_name)
        seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        img_info['ann'] = dict(seg_map=seg_map)
        img_dir=os.path.join(self.data_root,'data',video,'origin/')
        # ann_dir=os.path.join(self.data_root,'data',video,'mask/')
        ann_dir=os.path.join(self.data_root,'Detectron',self.mask_file,'data',video)

        # img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results, img_dir, ann_dir)
        return self.pipeline(results)

    def prepare_test_img2(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """
        video_imgname=self.img_all[idx]
        video, img_name=video_imgname[0], video_imgname[1]
        imglist = self.imgdic[video]
        img_index=imglist.index(img_name)
        
        this_step=[]
        this_steps=[]
        for dil in self.dilation:
            clip_index=img_index+dil
            if clip_index>=0 and clip_index<len(imglist):
                this_step.append(clip_index)
        this_step.append(img_index)
        
        if self.dilation==[-9,-6,-3]:
            if img_index==3:
                this_step=[0,1,2,3]
            elif img_index==4:
                this_step=[0,2,3,4]
            elif img_index==5:
                this_step=[0,2,4,5]
            elif img_index==6:
                this_step=[0,2,4,6]
            elif img_index==7:
                this_step=[0,3,5,7]
            elif img_index==8:
                this_step=[0,3,6,8]
                
        # print(idx,video_imgname,video,img_name,len(imglist),img_index,this_step)
        

        clips_img = []
        clips_target=[]
        clips_meta=[]
        results_all=[]

        img_info_clips, ann_info_clips, seg_fields_clips, img_prefix_clips, seg_prefix_clips, filename_clips=[],[],[],[],[],[]
        ori_filename_clips, img_clips, img_shape_clips, ori_shape_clips, pad_shape_clips=[],[],[],[],[]
        scale_factor_clips, img_norm_cfg_clips, gt_semantic_seg_clips=[],[],[]

        # for i in list(range(len(imglist[:15:1]))):
        
        # load farest frame
        len_s = len(imglist)
        tmp_far=0
        if (len_s-this_step[-1])>this_step[0]:
            tmp_far = len_s-1
            
        img_name=imglist[tmp_far]
        img_info=dict(filename=img_name)
        seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        img_info['ann'] = dict(seg_map=seg_map)
        ann_info=dict(seg_map=seg_map)
        img_dir=os.path.join(self.data_root,'data',video,'origin/')
        # ann_dir=os.path.join(self.data_root,'data',video,'mask/')
        ann_dir=os.path.join(self.data_root,'Detectron',self.mask_file,'data',video)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results, img_dir, ann_dir)
        self.pipeline_load(results)
        img_clips.append(results["img"]) 
                    
        for i in this_step:
            img_name=imglist[i]
            img_info=dict(filename=img_name)
            seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
            img_info['ann'] = dict(seg_map=seg_map)
            ann_info=dict(seg_map=seg_map)
            img_dir=os.path.join(self.data_root,'data',video,'origin/')
            # ann_dir=os.path.join(self.data_root,'data',video,'mask/')
            ann_dir=os.path.join(self.data_root,'Detectron',self.mask_file,'data',video)
            results = dict(img_info=img_info, ann_info=ann_info)
            self.pre_pipeline(results, img_dir, ann_dir)
            # clips_img.append(results['img'])
            # clips_target.append(results['gt_semantic_seg'])
            # clips_meta.append(results['img_metas'])
            self.pipeline_load(results)
            results_all.append(results)
            img_info_clips.append(results['img_info'])
            ann_info_clips.append(results['ann_info'])
            seg_fields_clips.append(results["seg_fields"])
            img_prefix_clips.append(results["img_prefix"])
            seg_prefix_clips.append(results["seg_prefix"])
            filename_clips.append(results["filename"])
            ori_filename_clips.append(results["ori_filename"])
            img_clips.append(results["img"]) 
            img_shape_clips.append(results["img_shape"])
            ori_shape_clips.append(results["ori_shape"])
            pad_shape_clips.append(results["pad_shape"])
            scale_factor_clips.append(results["scale_factor"])
            img_norm_cfg_clips.append(results["img_norm_cfg"])
        
        # if this_step[-1]<6:
        #     for i in this_step:
        #         img_name=imglist[i]
        #         img_info=dict(filename=img_name)
        #         seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        #         img_info['ann'] = dict(seg_map=seg_map)
        #         ann_info=dict(seg_map=seg_map)
        #         img_dir=os.path.join(self.data_root,'data',video,'origin/')
        #         ann_dir=os.path.join(self.data_root,'data',video,'mask/')
        #         results = dict(img_info=img_info, ann_info=ann_info)
        #         self.pre_pipeline(results, img_dir, ann_dir)
        #         # clips_img.append(results['img'])
        #         # clips_target.append(results['gt_semantic_seg'])
        #         # clips_meta.append(results['img_metas'])
        #         self.pipeline_load(results)
        #         results_all.append(results)
        #         img_info_clips.append(results['img_info'])
        #         ann_info_clips.append(results['ann_info'])
        #         seg_fields_clips.append(results["seg_fields"])
        #         img_prefix_clips.append(results["img_prefix"])
        #         seg_prefix_clips.append(results["seg_prefix"])
        #         filename_clips.append(results["filename"])
        #         ori_filename_clips.append(results["ori_filename"])
        #         img_clips.append(results["img"]) 
        #         img_shape_clips.append(results["img_shape"])
        #         ori_shape_clips.append(results["ori_shape"])
        #         pad_shape_clips.append(results["pad_shape"])
        #         scale_factor_clips.append(results["scale_factor"])
        #         img_norm_cfg_clips.append(results["img_norm_cfg"])
        # else:
        #     # print(max(0,this_step[-1]-15),this_step[-1])
        #     for i in range(max(0,this_step[-1]-13),this_step[-1]):
        #         # print('i am in',idx,this_step,i)
        #         img_name=imglist[i]
        #         img_info=dict(filename=img_name)
        #         seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        #         img_info['ann'] = dict(seg_map=seg_map)
        #         ann_info=dict(seg_map=seg_map)
        #         img_dir=os.path.join(self.data_root,'data',video,'origin/')
        #         ann_dir=os.path.join(self.data_root,'data',video,'mask/')
        #         results = dict(img_info=img_info, ann_info=ann_info)
        #         self.pre_pipeline(results, img_dir, ann_dir)
        #         # clips_img.append(results['img'])
        #         # clips_target.append(results['gt_semantic_seg'])
        #         # clips_meta.append(results['img_metas'])
        #         self.pipeline_load(results)
        #         results_all.append(results)
        #         img_info_clips.append(results['img_info'])
        #         ann_info_clips.append(results['ann_info'])
        #         seg_fields_clips.append(results["seg_fields"])
        #         img_prefix_clips.append(results["img_prefix"])
        #         seg_prefix_clips.append(results["seg_prefix"])
        #         filename_clips.append(results["filename"])
        #         ori_filename_clips.append(results["ori_filename"])
        #         img_clips.append(results["img"]) 
        #         img_shape_clips.append(results["img_shape"])
        #         ori_shape_clips.append(results["ori_shape"])
        #         pad_shape_clips.append(results["pad_shape"])
        #         scale_factor_clips.append(results["scale_factor"])
        #         img_norm_cfg_clips.append(results["img_norm_cfg"])
                # gt_semantic_seg_clips.append(results["gt_semantic_seg"])
                # for key, value in results.item():
                # print(results["seg_fields"])
                # exit()

        results_new=dict(img_info=img_info_clips[-1],ann_info=ann_info_clips[-1],seg_fields=seg_fields_clips[-1],
            img_prefix=img_prefix_clips[-1],seg_prefix=seg_prefix_clips[-1],
            filename=filename_clips[-1],ori_filename=ori_filename_clips[-1],img=img_clips,
            img_shape=img_shape_clips[-1],ori_shape=ori_shape_clips[-1],
            pad_shape=pad_shape_clips[-1],scale_factor=scale_factor_clips[-1],
            img_norm_cfg=img_norm_cfg_clips[-1])

        return self.pipeline_process(results_new)

        # img_info=dict(filename=img_name)
        # seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        # img_info['ann'] = dict(seg_map=seg_map)
        # img_dir=os.path.join(self.data_root,'data',video,'origin/')
        # ann_dir=os.path.join(self.data_root,'data',video,'mask/')

        # # img_info = self.img_infos[idx]
        # results = dict(img_info=img_info)
        # self.pre_pipeline(results, img_dir, ann_dir)
        # return self.pipeline(results)



    def format_results(self, results, save_path=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        # pass
        ## changed by guosun
        assert len(results)==len(self.img_all)
        palette_list=[]
        for kk in self.PALETTE:
            palette_list=palette_list+kk

        for ii in range(len(results)):
            result=results[ii]
            video_imgname=self.img_all[ii]
            video, img_name=video_imgname[0], video_imgname[1]
            if isinstance(result, str):
                result = np.load(result)
            save_path_ii=os.path.join(save_path,'result_submission',video,img_name.replace(self.img_suffix, self.seg_map_suffix))
            save_path_directory=os.path.dirname(save_path_ii)
            if not os.path.exists(save_path_directory):
                os.makedirs(save_path_directory)
            res = Image.fromarray(result.astype(np.uint8), mode='P')
            res.putpalette(palette_list)
            res.save(save_path_ii)

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""

        gt_seg_maps = []
        for video_imgname in self.img_all:
            video, img_name=video_imgname[0], video_imgname[1]
            seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
            # ann_dir=os.path.join(self.data_root,'data',video,'mask/')
            ann_dir=os.path.join(self.data_root,'Detectron',self.mask_file,'data',video)
            

            seg_map = osp.join(ann_dir, seg_map)
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
            gt_seg_maps.append(gt_seg_map)
            # print(seg_map,efficient_test,np.unique(gt_seg_map))

        # gt_seg_maps = []
        # for img_info in self.img_infos:
        #     seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
        #     if efficient_test:
        #         gt_seg_map = seg_map
        #     else:
        #         gt_seg_map = mmcv.imread(
        #             seg_map, flag='unchanged', backend='pillow')
        #     gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        # print(self.CLASSES)
        # exit()
        
        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
                         [i]] = summary_table_data[1][i] / 100.0
        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results

@DATASETS.register_module()
class CustomDataset_video2_vps(Dataset):
    """Custom dataset for video semantic segmentation. An example of file structure
    is as followed.

    return video clips instead of sepearate frames

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── video
        │   │   │   ├── origin
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── mask
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 dilation=[-4,-3,-2,-1],
                 clipnum=None,
                 video_dataset_list=["ASU-Mayo_Clinic", "CVC-ClinicDB-612", "CVC-ColonDB-300"]):
        # self.pipeline = Compose(pipeline)
        if split=='train':
            self.pipeline_load = Compose(pipeline[:2])
            self.pipeline_process = Compose(pipeline[2:])
        else:
            self.pipeline_load = Compose(pipeline[:1])
            self.pipeline_process = Compose(pipeline[1:])

        self.img_dir = ''
        self.img_suffix = img_suffix
        self.ann_dir = ''
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        self.dilation=dilation
        self.clipnum=clipnum

        # join paths if data_root is specified
        # if self.data_root is not None:
        #     if not osp.isabs(self.img_dir):
        #         self.img_dir = osp.join(self.data_root, self.img_dir)
        #     if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
        #         self.ann_dir = osp.join(self.data_root, self.ann_dir)
        #     if not (self.split is None or osp.isabs(self.split)):
        #         self.split = osp.join(self.data_root, self.split)

        # with open(os.path.join(self.data_root,self.split+'.txt')) as f:
        #     lines=f.readlines()
        #     self.videolists = [line[:-1] for line in lines]

        assert self.split in ['train', 'val']
        if self.split == 'train':
            self.path_res='Train'
        else:
            self.path_res=''

        self.videolists=[]
        self.imgdic={}
        self.img_all=[]

        if self.split == 'train':
            for video_name in video_dataset_list:
                video_root = os.path.join(data_root, video_name, self.path_res)
                cls_list = os.listdir(video_root)
                for cls in cls_list:
                    cls_newname=video_name+'___'+cls
                    cls_path = os.path.join(video_root, cls)
                    cls_img_path = os.path.join(cls_path, "Frame")
                    cls_label_path = os.path.join(cls_path, "GT")
                    tmp_list = os.listdir(cls_img_path)
                    tmp_list.sort()
                    self.imgdic[cls_newname]=tmp_list
                    self.img_all=self.img_all+[[cls_newname, img] for img in tmp_list]
                    self.videolists.append(cls_newname)
        else:
            for video_name in video_dataset_list:
                video_root = os.path.join(data_root, video_name, self.path_res)
                video_root = os.path.join(video_root, "Frame")
                cls_list = os.listdir(video_root)
                for cls in cls_list:
                    cls_newname=video_name+'___'+cls
                    cls_path = os.path.join(video_root, cls)
                    tmp_list = os.listdir(cls_path)
                    tmp_list.sort()
                    self.imgdic[cls_newname]=tmp_list
                    self.img_all=self.img_all+[[cls_newname, img] for img in tmp_list]
                    self.videolists.append(cls_newname)
        
        # self.imgdic={}
        # self.img_all=[]
        # for video in self.videolists:
        #     v_path = os.path.join(self.data_root,'data',video,'origin')
        #     imglist = sorted(os.listdir(v_path))
        #     self.imgdic[video]=imglist
        #     self.img_all=self.img_all+[[video, img] for img in imglist]

        # self.flip_video=False
        self.flip_video=True
        print("flip video: ",self.flip_video)
        # print(self.label_map)
        # print(self.CLASSES, self.PALETTE)
        # exit()

        # if self.split=='train':
            # self.img_all=self.img_all[:500]
        # load annotations
        # self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
        #                                        self.ann_dir,
        #                                        self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        # return len(self.img_infos)
        if self.split=='train':
            return len(self.videolists)
        else:
            return len(self.img_all)
        # return len(self.img_all)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results, img_dir,  ann_dir):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = img_dir
        results['seg_prefix'] = ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img2(idx)
        else:
            return self.prepare_train_img2(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        # video  = self.videolists[idx]
        # imglist = self.imgdic[video]
        # imglist_s = imglist[-self.dilation[0]:]
        # if len(imglist_s)<1:
        #     return None
        # idx = np.random.choice(list(range(len(imglist_s))))-self.dilation[0]
        # this_step=[]
        # for dil in self.dilation:
        #     this_step.append(idx+dil)
        # this_step.append(idx)

        # clips_img = []
        # clips_target=[]
        # clips_meta=[]
        # for i in this_step:
        #     img_name=imglist[i]
        #     img_info=dict(filename=img_name)
        #     seg_map = img_name.replace(img_suffix, seg_map_suffix)
        #     ann_info=dict(seg_map=seg_map)
        #     self.img_dir=os.path.join(self.dataroot,'data',video,'origin/')
        #     self.ann_dir=os.path.join(self.dataroot,'data',video,'mask/')
        #     results = dict(img_info=img_info, ann_info=ann_info)
        #     self.pre_pipeline(results)
        #     clips_img.append(results['img'])
        #     clips_target.append(results['gt_semantic_seg'])
        #     clips_meta.append(results['img_metas'])

        # return dict(clips_img=clips_img,clips_target=clips_target,clips_meta=clips_meta)

        video_imgname=self.img_all[idx]
        video, img_name=video_imgname[0], video_imgname[1]
        img_info=dict(filename=img_name)
        seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        img_info['ann'] = dict(seg_map=seg_map)
        ann_info=dict(seg_map=seg_map)
        img_dir=os.path.join(self.data_root,'data',video,'origin/')
        ann_dir=os.path.join(self.data_root,'data',video,'mask/')
        results = dict(img_info=img_info, ann_info=ann_info)

        # img_info = self.img_infos[idx]
        # ann_info = self.get_ann_info(idx)
        # results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results, img_dir, ann_dir)
        return self.pipeline(results)

    def prepare_train_img2(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        video  = self.videolists[idx]
        imglist = self.imgdic[video]
        ## inverse video with probability of 0.5
        if self.flip_video:
            # print("here")
            if random.random()<0.5:
                imglist=imglist[::-1]
        imglist_s = imglist[-self.dilation[0]:]
        if len(imglist_s)<1:
            return None
        idx = np.random.choice(list(range(len(imglist_s))))-self.dilation[0]
        this_step=[]
        for dil in self.dilation:
            this_step.append(idx+dil)
        this_step.append(idx)

        clips_img = []
        clips_target=[]
        clips_meta=[]
        results_all=[]

        img_info_clips, ann_info_clips, seg_fields_clips, img_prefix_clips, seg_prefix_clips, filename_clips=[],[],[],[],[],[]
        ori_filename_clips, img_clips, img_shape_clips, ori_shape_clips, pad_shape_clips=[],[],[],[],[]
        scale_factor_clips, img_norm_cfg_clips, gt_semantic_seg_clips=[],[],[]

        print('2',len(imglist))

        for i in imglist:
        # for i in this_step:
            img_name=imglist[i]
            img_info=dict(filename=img_name)
            seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
            img_info['ann'] = dict(seg_map=seg_map)
            ann_info=dict(seg_map=seg_map)
            # img_dir=os.path.join(self.data_root,'data',video,'origin/')
            # ann_dir=os.path.join(self.data_root,'data',video,'mask/')
            if self.split =='train':
                img_dir = os.path.join(self.data_root, video.replace('___', '/%s/'%(self.path_res)), "Frame")
                ann_dir = os.path.join(self.data_root, video.replace('___', '/%s/'%(self.path_res)), "GT")
            else:
                img_dir = os.path.join(self.data_root, video)
                ann_dir = img_dir.replace('/Frame/', '/GT/')
            results = dict(img_info=img_info, ann_info=ann_info)
            self.pre_pipeline(results, img_dir, ann_dir)
            # clips_img.append(results['img'])
            # clips_target.append(results['gt_semantic_seg'])
            # clips_meta.append(results['img_metas'])
            self.pipeline_load(results)
            results_all.append(results)
            img_info_clips.append(results['img_info'])
            ann_info_clips.append(results['ann_info'])
            seg_fields_clips.append(results["seg_fields"])
            img_prefix_clips.append(results["img_prefix"])
            seg_prefix_clips.append(results["seg_prefix"])
            filename_clips.append(results["filename"])
            ori_filename_clips.append(results["ori_filename"])
            img_clips.append(results["img"]) 
            img_shape_clips.append(results["img_shape"])
            ori_shape_clips.append(results["ori_shape"])
            pad_shape_clips.append(results["pad_shape"])
            scale_factor_clips.append(results["scale_factor"])
            img_norm_cfg_clips.append(results["img_norm_cfg"])
            gt_semantic_seg_clips.append(results["gt_semantic_seg"])
            # for key, value in results.item():
            # print(results["seg_fields"])
            # exit()

        results_new=dict(img_info=img_info_clips[-1],ann_info=ann_info_clips[-1],seg_fields=seg_fields_clips[-1],
            img_prefix=img_prefix_clips[-1],seg_prefix=seg_prefix_clips[-1],
            filename=filename_clips[-1],ori_filename=ori_filename_clips[-1],img=img_clips,
            img_shape=img_shape_clips[-1],ori_shape=ori_shape_clips[-1],
            pad_shape=pad_shape_clips[-1],scale_factor=scale_factor_clips[-1],
            img_norm_cfg=img_norm_cfg_clips[-1],gt_semantic_seg=gt_semantic_seg_clips)

        # self.pipeline_process(results_new)

        # print(results_new.keys())

        # exit()

        return self.pipeline_process(results_new)

        # img_info = self.img_infos[idx]
        # ann_info = self.get_ann_info(idx)
        # results = dict(img_info=img_info, ann_info=ann_info)
        # self.pre_pipeline(results)
        # return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """
        video_imgname=self.img_all[idx]
        video, img_name=video_imgname[0], video_imgname[1]
        img_info=dict(filename=img_name)
        seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        img_info['ann'] = dict(seg_map=seg_map)
        img_dir=os.path.join(self.data_root,'data',video,'origin/')
        ann_dir=os.path.join(self.data_root,'data',video,'mask/')

        # img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results, img_dir, ann_dir)
        return self.pipeline(results)

    def prepare_test_img2(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """
        video_imgname=self.img_all[idx]
        video, img_name=video_imgname[0], video_imgname[1]
        imglist = self.imgdic[video]
        img_index=imglist.index(img_name)
        this_step=[]
        for dil in self.dilation:
            clip_index=img_index+dil
            if clip_index>=0 and clip_index<len(imglist):
                this_step.append(clip_index)
        this_step.append(img_index)

        if self.dilation==[-9,-6,-3]:
            if img_index==3:
                this_step=[0,1,2,3]
            elif img_index==4:
                this_step=[0,2,3,4]
            elif img_index==5:
                this_step=[0,2,4,5]
            elif img_index==6:
                this_step=[0,2,4,6]
            elif img_index==7:
                this_step=[0,3,5,7]
            elif img_index==8:
                this_step=[0,3,6,8]

        clips_img = []
        clips_target=[]
        clips_meta=[]
        results_all=[]

        img_info_clips, ann_info_clips, seg_fields_clips, img_prefix_clips, seg_prefix_clips, filename_clips=[],[],[],[],[],[]
        ori_filename_clips, img_clips, img_shape_clips, ori_shape_clips, pad_shape_clips=[],[],[],[],[]
        scale_factor_clips, img_norm_cfg_clips, gt_semantic_seg_clips=[],[],[]

        for i in this_step:
            img_name=imglist[i]
            img_info=dict(filename=img_name)
            seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
            img_info['ann'] = dict(seg_map=seg_map)
            ann_info=dict(seg_map=seg_map)
            # img_dir=os.path.join(self.data_root,'data',video,'origin/')
            # ann_dir=os.path.join(self.data_root,'data',video,'mask/')
            img_dir=os.path.join(self.data_root, video.replace('___','/Frame/'))
            ann_dir=img_dir.replace('/Frame/', '/GT/')
            results = dict(img_info=img_info, ann_info=ann_info)
            self.pre_pipeline(results, img_dir, ann_dir)
            # clips_img.append(results['img'])
            # clips_target.append(results['gt_semantic_seg'])
            # clips_meta.append(results['img_metas'])
            self.pipeline_load(results)
            results_all.append(results)
            img_info_clips.append(results['img_info'])
            ann_info_clips.append(results['ann_info'])
            seg_fields_clips.append(results["seg_fields"])
            img_prefix_clips.append(results["img_prefix"])
            seg_prefix_clips.append(results["seg_prefix"])
            filename_clips.append(results["filename"])
            ori_filename_clips.append(results["ori_filename"])
            img_clips.append(results["img"]) 
            img_shape_clips.append(results["img_shape"])
            ori_shape_clips.append(results["ori_shape"])
            pad_shape_clips.append(results["pad_shape"])
            scale_factor_clips.append(results["scale_factor"])
            img_norm_cfg_clips.append(results["img_norm_cfg"])
            # gt_semantic_seg_clips.append(results["gt_semantic_seg"])
            # for key, value in results.item():
            # print(results["seg_fields"])
            # exit()

        results_new=dict(img_info=img_info_clips[-1],ann_info=ann_info_clips[-1],seg_fields=seg_fields_clips[-1],
            img_prefix=img_prefix_clips[-1],seg_prefix=seg_prefix_clips[-1],
            filename=filename_clips[-1],ori_filename=ori_filename_clips[-1],img=img_clips,
            img_shape=img_shape_clips[-1],ori_shape=ori_shape_clips[-1],
            pad_shape=pad_shape_clips[-1],scale_factor=scale_factor_clips[-1],
            img_norm_cfg=img_norm_cfg_clips[-1])

        # self.pipeline_process(results_new)

        # print(results_new.keys())

        # exit()

        return self.pipeline_process(results_new)

        # img_info=dict(filename=img_name)
        # seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
        # img_info['ann'] = dict(seg_map=seg_map)
        # img_dir=os.path.join(self.data_root,'data',video,'origin/')
        # ann_dir=os.path.join(self.data_root,'data',video,'mask/')

        # # img_info = self.img_infos[idx]
        # results = dict(img_info=img_info)
        # self.pre_pipeline(results, img_dir, ann_dir)
        # return self.pipeline(results)



    def format_results(self, results, save_path=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        # pass
        ## changed by guosun
        assert len(results)==len(self.img_all)
        palette_list=[]
        for kk in self.PALETTE:
            palette_list=palette_list+kk

        for ii in range(len(results)):
            result=results[ii]
            video_imgname=self.img_all[ii]
            video, img_name=video_imgname[0], video_imgname[1]
            # print(result)
            if isinstance(result, str):
                result = np.load(result)*255
                result=result.squeeze(0)
                print(result.dtype,result.min(),result.max(),result.shape)
            save_path_ii=os.path.join(save_path,'result_submission',video.replace('___','/Pred/'),img_name.replace(self.img_suffix, self.seg_map_suffix))
            save_path_directory=os.path.dirname(save_path_ii)
            if not os.path.exists(save_path_directory):
                os.makedirs(save_path_directory)
            res = Image.fromarray(result.astype(np.uint8), mode='P')
            res.putpalette(palette_list)
            res.save(save_path_ii)

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""

        gt_seg_maps = []
        for video_imgname in self.img_all:
            video, img_name=video_imgname[0], video_imgname[1]
            seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
            ann_dir=os.path.join(self.data_root,'data',video,'mask/')

            seg_map = osp.join(ann_dir, seg_map)
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
            gt_seg_maps.append(gt_seg_map)

        # gt_seg_maps = []
        # for img_info in self.img_infos:
        #     seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
        #     if efficient_test:
        #         gt_seg_map = seg_map
        #     else:
        #         gt_seg_map = mmcv.imread(
        #             seg_map, flag='unchanged', backend='pillow')
        #     gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        print('nn',num_classes)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
                         [i]] = summary_table_data[1][i] / 100.0
        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results

