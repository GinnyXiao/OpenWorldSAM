# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Boxes, Instances
from torchvision import transforms

# tokenizing the prompts
from transformers import AutoTokenizer

__all__ = ["OpenWorldSAM2PanopticDatasetMapper"]


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


def sam_preprocess(x: np.ndarray) -> torch.Tensor:
    """
    Preprocess for the Segment Anything Model (SAM), including scaling, normalization, and padding.
    """
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024

    x = torch.as_tensor(np.ascontiguousarray(x.transpose(2, 0, 1)))
    x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
    x = (x - pixel_mean) / pixel_std

    return x

def beit3_preprocess(x: np.ndarray) -> torch.Tensor:
    """
    Preprocess for BEIT-3 model.
    """
    img_size = 224
    beit_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=3, antialias=None),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return beit_preprocess(np.array(x))

def get_class_name_from_id_hack(metadata, class_id):
    """Get class name from class_id using metadata"""
    for k, v in metadata.stuff_dataset_id_to_contiguous_id.items():
        if v == class_id:
            return metadata.stuff_classes[v]

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


class OpenWorldSAM2PanopticDatasetMapper:
    """
    A callable dataset mapper for OpenWorldSAM2 panoptic segmentation.
    
    Unlike the original mapper, this one returns instances directly without grouping by category.
    This approach aligns more closely with the instance dataset mapper's handling of annotations.
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
        tokenizer,
        dataset_name
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
        self.dataset_name = dataset_name

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
            "tokenizer": tokenizer,
            "dataset_name": dataset_name
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

        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

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
            
            # Return all instances directly instead of grouping by category
            dataset_dict["instances"] = instances

            # Generate prompt from unique categories
            unique_categories = list(set(instances.gt_classes.tolist()))
            unique_categories.sort()  # Ensure consistent ordering
            
            # More aggressive limit for prompts to prevent CUDA OOM
            # For images with large number of categories, limit more strictly
            max_prompts = 20  # Default maximum number of prompts
            
            if len(unique_categories) > max_prompts:
                # Sample prompts randomly
                import random
                random.seed(42)  # For reproducibility
                sampled_indices = random.sample(range(len(unique_categories)), max_prompts)
                sampled_categories = [unique_categories[i] for i in sampled_indices]
                
                # Filter instances to only keep those belonging to sampled categories
                keep_mask = torch.zeros_like(instances.gt_classes, dtype=torch.bool)
                for cat_id in sampled_categories:
                    keep_mask |= (instances.gt_classes == cat_id)
                
                instances = Instances(
                    image_shape,
                    gt_classes=instances.gt_classes[keep_mask],
                    gt_masks=instances.gt_masks[keep_mask],
                    gt_boxes=instances.gt_boxes[keep_mask]
                )
                
                unique_categories = sampled_categories
                dataset_dict["instances"] = instances
            
            # Create class name list as prompts
            class_names = []
            for category_id in unique_categories:
                if "ade20k" in self.dataset_name or "bdd" in self.dataset_name or "scannet" in self.dataset_name:
                    # class_name = get_class_name_from_id(self.metadata, category_id, self.thing_ids)
                    class_name = get_class_name_from_id_hack(self.metadata, category_id)
                else:
                    class_name = get_class_name_from_id(self.metadata, category_id, self.thing_ids)
                if class_name is not None:
                    class_names.append(class_name)
            
            dataset_dict["prompt"] = [text.replace('-other','').replace('-merged','').replace('-stuff','') for text in class_names]
            dataset_dict["unique_categories"] = unique_categories
        
        return dataset_dict 