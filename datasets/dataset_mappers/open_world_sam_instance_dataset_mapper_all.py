import copy
import logging

import numpy as np
import torch
import random

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances, PolygonMasks, Boxes, polygons_to_bitmask
from .transforms import ResizeLongestSide, Resize
import torch.nn.functional as F
from torchvision import transforms

from pycocotools import mask as coco_mask
import pycocotools.mask as mask_util

# tokenizing the prompts
from transformers import AutoTokenizer

__all__ = ["OpenWorldSAM2InstanceDatasetMapperAll"]


def filter_empty_instances_by_box(
        instances, by_box=True, by_mask=False, box_threshold=1e-5, return_mask=False
):
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    # TODO: can also filter visible keypoints

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x
    if return_mask:
        return instances[m], m
    return instances[m]

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def sam_preprocess(
        x: np.ndarray,
        pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
        pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
        img_size=1024) -> torch.Tensor:
    '''
    preprocess of Segment Anything Model, including scaling, normalization and padding.
    input: ndarray
    output: torch.Tensor
    '''
    assert img_size == 1024, \
        " SAM receive images of size 1024^2, don't change this setting unless you're sure that your employed model works well with another size."

    x = torch.as_tensor(np.ascontiguousarray(x.transpose(2, 0, 1)))
    x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
    x = (x - pixel_mean) / pixel_std

    return x


def beit3_preprocess(x: np.ndarray, img_size=224) -> torch.Tensor:
    '''
    preprocess for BEIT-3 model.
    input: ndarray
    output: torch.Tensor
    '''
    beit_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=3, antialias=None),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return beit_preprocess(np.array(x))

def get_categoryID_to_text_mapping(cfg, dataset_index=0):
    dataset_name = cfg.DATASETS.TRAIN[dataset_index]
    metadata = MetadataCatalog.get(dataset_name)
    if dataset_name == "coco_2017_train" or dataset_name == "coco_2017_val":
        thing_classes = metadata.get("thing_classes")
        return thing_classes
    elif dataset_name == "ade20k_instance_train" or dataset_name == "ade20k_instance_val":
        thing_classes = metadata.get("thing_classes")
        return thing_classes
    elif dataset_name == "lvis_v1_train+coco_panoptic_separated":
        return metadata.get("stuff_classes")
    elif dataset_name == "lvis_v1_train+coco":
        return metadata.get("thing_classes")
    else:
        return None

def build_tokenizer(cfg):
    # tokenizer
    tokenizer_config = cfg.MODEL.OpenWorldSAM2.TOKENIZER_CONFIG
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config, padding_side="right", use_fast=False)
    return tokenizer

def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    # assert is_train, "Only support training augmentation"

    # EVF-SAM  no data augmentation
    augmentation = []

    return augmentation


class OpenWorldSAM2InstanceDatasetMapperAll:
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
            tfm_gens,
            image_format,
            text_classes=None,
            tokenizer=None
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOInstanceNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
        self.text_classes = text_classes
        self.tokenizer = tokenizer

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        # get metadata
        text_classes = get_categoryID_to_text_mapping(cfg)

        # tokenizer
        tokenizer = build_tokenizer(cfg)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "text_classes": text_classes,
            "tokenizer": tokenizer
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # Get padding mask
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Prepare images for SAM2 and BEIT-3
        dataset_dict["image"] = sam_preprocess(image)
        dataset_dict["evf_image"] = beit3_preprocess(image)
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        # Set a default prompt immediately
        dataset_dict["prompt"] = ["object"]
        dataset_dict["unique_categories"] = [0]

        # Add all COCO categories for negative sampling
        if self.text_classes is not None:
            dataset_dict["all_coco_categories"] = self.text_classes

        if "annotations" in dataset_dict:
            # Transform and process annotations
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict["annotations"]
                if obj.get("iscrowd", 0) == 0
            ]

            if len(annos):
                assert "segmentation" in annos[0]
            segms = [obj["segmentation"] for obj in annos]
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image.shape[:2]))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a binary segmentation mask "
                        " in a 2D numpy array of shape HxW.".format(type(segm))
                    )

            # Pad image and segmentation label here!
            masks = [torch.from_numpy(np.ascontiguousarray(x)) for x in masks]

            classes = [int(obj["category_id"]) for obj in annos]
            classes = torch.tensor(classes, dtype=torch.int64)

            # Prepare per-category binary masks
            instances = Instances(image_shape)
            instances.gt_classes = classes

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, image.shape[-2], image.shape[-1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(torch.stack(masks))
                instances.gt_boxes = masks.get_bounding_boxes()
                instances.gt_masks = masks.tensor

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

            # Create ordered prompts and instances
            prompts = [self.text_classes[cat_id] for cat_id in sorted(category_to_instances.keys())]
            # print("prompts:", prompts)
            dataset_dict["prompt"] = prompts

            # if not self.is_train:
            #     # USER: Modify this if you want to keep them for some reason.
            #     # for test set, do not return ground truth instances
            #     return dataset_dict

            ordered_instances = []
            for cat_id in sorted(category_to_instances.keys()):
                indices = category_to_instances[cat_id]
                selected_instances = instances[indices]  # Select instances for this category
                ordered_instances.append(selected_instances)

            dataset_dict["instances"] = ordered_instances
            dataset_dict["unique_categories"] = sorted(category_to_instances.keys())
            # print(ordered_instances)

        return dataset_dict