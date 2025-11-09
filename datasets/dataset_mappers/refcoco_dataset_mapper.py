# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import random

import scipy.io
import numpy as np
import torch
from PIL import Image

from pycocotools import mask
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from detectron2.config import configurable
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

__all__ = ["RefCOCODatasetMapper"]

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


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    augmentation = []
    return augmentation


# This is specifically designed for the COCO dataset.
class RefCOCODatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        metadata,
        tokenizer,
        dataset_name
    ):
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
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
        
        tokenizer_config = cfg.MODEL.OpenWorldSAM2.TOKENIZER_CONFIG
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config, padding_side="right", use_fast=False)

        return {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "metadata": meta,
            "tokenizer": tokenizer,
            "dataset_name": dataset_name
        }

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        file_name = dataset_dict['file_name']

        image = utils.read_image(file_name, format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # Original image format for height/width
        image_shape = image.shape[:2]  # h, w

        # Get padding mask for transformer's attention
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

        grounding_anno = dataset_dict['grounding_info']
        assert len(grounding_anno) > 0
        masks_grd = []
        texts_grd = []
        boxes_grd = []
        for ann in grounding_anno:
            rle = mask.frPyObjects(
                ann['segmentation'], dataset_dict['height'], dataset_dict['width'])
            m = mask.decode(rle)
            # sometimes there are multiple binary map (corresponding to multiple segs)
            m = np.sum(m, axis=2)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks_grd += [m]
            texts_grd.append([x['raw'].lower() for x in ann['sentences']])
            boxes_grd.append(ann['bbox']) # xywh
        masks_grd = torch.from_numpy(np.stack(masks_grd))
        boxes_grd = torch.tensor(boxes_grd)

        groundings = {'masks': masks_grd, 'texts': texts_grd, 'boxes': boxes_grd}
        dataset_dict["groundings"] = groundings

        # Create unique labels for each text prompt
        # unique_texts = list(set([text for texts in texts_grd for text in texts])) if not self.is_train else texts_grd
        
        
        # Set prompts and labels
        if self.is_train:
            # For training, randomly select one expression from each item in texts_grd
            dataset_dict["prompt"] = [random.choice(x) for x in texts_grd]
            dummy_classes = list(range(len(texts_grd)))
            dataset_dict["unique_categories"] = dummy_classes

            # Group instances by category
            grouped_instances = []
            for class_id in dummy_classes:
                # Create a new Instances object for this category
                category_instances = Instances(image_shape)
                # Add the mask, box and class for this instance
                category_instances.gt_masks = masks_grd[class_id:class_id+1]  # Keep as 2D tensor
                category_instances.gt_boxes = boxes_grd[class_id:class_id+1]  # Keep as 2D tensor
                category_instances.gt_classes = torch.tensor([class_id], dtype=torch.int64)
                grouped_instances.append(category_instances)

            # Return list of instances grouped by category
            dataset_dict["instances"] = grouped_instances
        else:
            dataset_dict["prompt"] = [x[0] for x in texts_grd] # take the first referring expression: follow X-Decoder
            dummy_classes = list(range(len(texts_grd)))
            dataset_dict["unique_categories"] = dummy_classes

            # Create instances with proper class labels
            instances = Instances(image_shape)
            instances.gt_masks = masks_grd
            instances.gt_boxes = boxes_grd
            instances.gt_classes = torch.tensor(dummy_classes, dtype=torch.int64)

            # instances = filter_empty_instances_by_box(instances)

            # Return all instances directly instead of grouping by category
            dataset_dict["instances"] = instances
        
        return dataset_dict