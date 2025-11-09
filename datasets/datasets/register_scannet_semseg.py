# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
import numpy as np
import os
import glob
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

from utils.constants import SCAN_40, SCAN_20

__all__ = ["load_scannet_instances", "register_scannet_context"]

def load_scannet_instances(name: str, dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load ScanNet annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    annpath = os.path.join(dirname, split + '.txt')
    dataroot = os.path.join(dirname, 'scannet_frames_25k')
    with open(annpath, 'r') as fr:
        pairs = fr.read().splitlines()
    img_paths, lb_paths = [], []
    for pair in pairs:
        imgpth, lbpth = pair.split(',')
        if name == "scannet_41_val_seg":
            lbpth = lbpth.replace('label20', 'label')
        img_paths.append(os.path.join(dataroot, imgpth))
        lb_paths.append(os.path.join(dataroot, lbpth))

    assert len(img_paths) == len(lb_paths)
    dataset_dicts = []
    for (img_path, gt_path) in zip(img_paths, lb_paths):
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path
        record["image_id"] = gt_path.split('/')[-3] + gt_path.split('/')[-1].split('.')[0],
        dataset_dicts.append(record)

    return dataset_dicts


def register_scannet_context(name, dirname, split, class_names=SCAN_20):
    if name == "scannet_21_val_seg":
        class_names = SCAN_20
    elif name == "scannet_41_val_seg":
        class_names = ["background"] + SCAN_40
    class_names = list(class_names)
    DatasetCatalog.register(name, lambda: load_scannet_instances(name, dirname, split, class_names))
    
    # Create mapping from dataset ID to contiguous ID
    stuff_dataset_id_to_contiguous_id = {i: i for i in range(len(class_names))}

    if name == "scannet_41_val_seg":

        MetadataCatalog.get(name).set(
            stuff_classes=class_names,
            dirname=dirname,
            split=split,
            ignore_label=0,
            thing_dataset_id_to_contiguous_id={},
            stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
            # class_offset=1,
            class_offset=0,
            keep_sem_bgd=False
        )
    elif name == "scannet_21_val_seg":
        MetadataCatalog.get(name).set(
            stuff_classes=class_names,
            dirname=dirname,
            split=split,
            ignore_label=[255],
            thing_dataset_id_to_contiguous_id={},
            stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
            # class_offset=1,
            class_offset=0,
            keep_sem_bgd=False
        )


def register_all_sunrgbd_seg(root):
    SPLITS = [
            ("scannet_21_val_seg", "scannet", "val"),
            ("scannet_41_val_seg", "scannet", "val"),
        ]
        
    for name, dirname, split in SPLITS:
        register_scannet_context(name, os.path.join(root, dirname), split)
        MetadataCatalog.get(name).evaluator_type = "sem_seg"


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_sunrgbd_seg(_root)