# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
import torch
import random
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Boxes, Instances
import detectron2.data.transforms as T
from torchvision import transforms
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

# tokenizing the prompts
from transformers import AutoTokenizer

__all__ = ["OpenWorldSAM2PanopticDatasetMapperAll"]


def filter_empty_instances_by_box(
        instances, by_box=True, by_mask=False, box_threshold=1e-5, return_mask=False
):
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x
    if return_mask:
        return instances[m], m
    return instances[m]


def sam_preprocess(
        x: np.ndarray,
        pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
        pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
        img_size=1024) -> torch.Tensor:
    assert img_size == 1024
    x = torch.as_tensor(np.ascontiguousarray(x.transpose(2, 0, 1)))
    x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
    x = (x - pixel_mean) / pixel_std
    return x

def beit3_preprocess(x: np.ndarray, img_size=224) -> torch.Tensor:
    beit_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=3, antialias=None),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return beit_preprocess(np.array(x))


def get_class_name_from_id(metadata, class_id, thing_ids):
    """Get class name from class_id using metadata"""
    if class_id in thing_ids:
        # It's a thing class
        for k, v in metadata.thing_dataset_id_to_contiguous_id.items():
            if v == class_id:
                return metadata.thing_classes[v]
    else:
        # It's a stuff class
        for k, v in metadata.stuff_dataset_id_to_contiguous_id.items():
            if v == class_id:
                return metadata.stuff_classes[v]
    return None


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    augmentation = []
    return augmentation


class OpenWorldSAM2PanopticDatasetMapperAll:
    """
    A callable dataset mapper for OpenWorldSAM2 panoptic segmentation.
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        label_divisor,
        thing_ids,
        stuff_ids,
        metadata,
        tokenizer
    ):
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.label_divisor = label_divisor
        self.thing_ids = thing_ids
        self.stuff_ids = stuff_ids
        self.metadata = metadata
        self.tokenizer = tokenizer

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        augs = build_transform_gen(cfg, is_train)

        dataset_name = cfg.DATASETS.TRAIN[0] if is_train else cfg.DATASETS.TEST[0]
        meta = MetadataCatalog.get(dataset_name)
        
        ignore_label = meta.ignore_label if hasattr(meta, "ignore_label") else 255
        label_divisor = meta.label_divisor if hasattr(meta, "label_divisor") else 1000
        thing_ids = list(meta.thing_dataset_id_to_contiguous_id.values())
        stuff_ids = list(meta.stuff_dataset_id_to_contiguous_id.values())
        
        tokenizer_config = cfg.MODEL.OpenWorldSAM2.TOKENIZER_CONFIG
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config, padding_side="right", use_fast=False)

        return {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY if hasattr(cfg.INPUT, "SIZE_DIVISIBILITY") else 0,
            "label_divisor": label_divisor,
            "thing_ids": thing_ids,
            "stuff_ids": stuff_ids,
            "metadata": meta,
            "tokenizer": tokenizer
        }

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that OpenWorldSAM2 can consume
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # Get padding mask for transformer's attention
        padding_mask = np.ones(image.shape[:2])

        # Panoptic segmentation format
        if "pan_seg_file_name" not in dataset_dict:
            raise ValueError(f"Cannot find 'pan_seg_file_name' for panoptic segmentation: {dataset_dict['file_name']}")

        pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"))

        padding_mask = np.ones(image.shape[:2])
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~padding_mask.astype(bool)

        # Apply SAM preprocessing
        dataset_dict["image"] = sam_preprocess(image)
        dataset_dict["evf_image"] = beit3_preprocess(image)
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        
        # Set a default prompt immediately
        dataset_dict["prompt"] = ["object"]
        dataset_dict["unique_categories"] = [0]
        
        # Original image format for height/width
        image_shape = image.shape[:2]  # h, w

        from panopticapi.utils import rgb2id

        pan_seg_gt = rgb2id(pan_seg_gt)

        instances = Instances(image_shape)
        classes = []
        masks = []
        segments_info = dataset_dict["segments_info"]
        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if not segment_info["iscrowd"]:
                classes.append(class_id)
                masks.append(pan_seg_gt == segment_info["id"])

        is_things = [COCO_CATEGORIES[idx]['isthing'] for idx in classes]
        classes = np.array(classes)
        is_things = np.array(is_things)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        instances.is_thing = torch.tensor(is_things, dtype=torch.bool)

        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            dataset_dict["instances"] = instances
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor
            instances.gt_boxes = masks.get_bounding_boxes()
            
            instances = filter_empty_instances_by_box(instances)

            # Group instances by category
            category_to_instances = {}
            for i, category_id in enumerate(instances.gt_classes.tolist()):
                if category_id not in category_to_instances:
                    category_to_instances[category_id] = []
                category_to_instances[category_id].append(i)

            # Sample at most 6 unique categories to prevent memory explosion
            unique_categories = list(category_to_instances.keys())
            if len(unique_categories) > 6:
                sampled_categories = random.sample(unique_categories, 6)
                # Filter category_to_instances to only include sampled categories
                category_to_instances = {k: v for k, v in category_to_instances.items() if k in sampled_categories}
            else:
                sampled_categories = unique_categories

            # Get class names for each sampled category
            class_names = []
            for category_id in sorted(category_to_instances.keys()):
                class_name = get_class_name_from_id(self.metadata, category_id, self.thing_ids)
                if class_name is not None:
                    class_names.append(class_name)
                
            # Create ordered prompts and instances
            dataset_dict["prompt"] = [text.replace('-other','').replace('-merged','').replace('-stuff','') for text in class_names]
            dataset_dict["unique_categories"] = sorted(category_to_instances.keys())
            
            # Create ordered instances by category
            ordered_instances = []
            for cat_id in sorted(category_to_instances.keys()):
                indices = category_to_instances[cat_id]
                selected_instances = instances[indices]  # Select instances for this category
                ordered_instances.append(selected_instances)
                
            dataset_dict["instances"] = ordered_instances

        
        return dataset_dict