import copy
import itertools
import logging
import os
import warnings
warnings.filterwarnings("ignore")

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
from tqdm import tqdm
from torch.utils.data import Subset
import random

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetCatalog, build_detection_test_loader
from detectron2.data.samplers import RandomSubsetTrainingSampler
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    verify_results,
)
import cv2
import matplotlib.pyplot as plt
import weakref

from datasets import (
    OpenWorldSAM2InstanceDatasetMapper,
    OpenWorldSAM2InstanceDatasetMapperAll,
    OpenWorldSAM2PanopticDatasetMapper,
    OpenWorldSAM2PanopticDatasetMapperAll,
    ScanNetPanoDatasetMapper,
    OpenWorldSAM2SemanticDatasetMapper,
    RefCOCODatasetMapper,
)

from evaluation import (
    # InstanceSegEvaluator,
    COCOPanopticEvaluator,
    SemSegEvaluator,
    GroundingEvaluator
)

from model import (
    add_open_world_sam2_config,
)

import random
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer
)

import numpy as np

# Add imports for our new OpenWorldSAM2WithPaliGemma model and mapper
# from datasets.dataset_mappers.open_world_sam_panoptic_dataset_mapper_paligemma import OpenWorldSAM2PanopticDatasetMapperPaliGemma

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        logger.info("Model on device:\n{}".format(model.device))
        model.print_trainable_parameters()
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)
        lr_scheduler = self.build_lr_scheduler(cfg, optimizer)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # add model EMA
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        # kwargs.update(model_ema.may_get_ema_checkpointer(cfg, model)) TODO: release ema training for large models
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        print("calling build_evaluator")
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            print("output_folder:", output_folder)
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        print("evaluator_type:", evaluator_type)
        # semantic segmentation
        if evaluator_type in ["sem_seg", ]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "scannet_panoptic_seg"
        ]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))

        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.OpenWorldSAM2.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.OpenWorldSAM2.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))

        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.OpenWorldSAM2.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))

        # RefCOCO
        if evaluator_type in ["grounding_refcoco"]:
            evaluator_list.append(GroundingEvaluator(dataset_name))

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if dataset_name in ["coco_2017_val", "ade20k_instance_val"]:
            mapper = OpenWorldSAM2InstanceDatasetMapper(cfg, is_train=False)
        elif dataset_name in ["coco_2017_val_panoptic_with_sem_seg", "ade20k_panoptic_val"]:
            mapper = OpenWorldSAM2PanopticDatasetMapper(cfg, is_train=False)
        elif dataset_name in ["scannet_21_panoptic_val"]:
            mapper = ScanNetPanoDatasetMapper(cfg, is_train=False)
        elif dataset_name in ["ade20k_full_sem_seg_val", "pascal_context_459_sem_seg_val",
                              "pascal_context_59_sem_seg_val", "pascalvoc20_sem_seg_val",
                              "sunrgbd_37_val_seg", "scannet_21_val_seg", "scannet_41_val_seg"]:
            mapper = OpenWorldSAM2SemanticDatasetMapper(cfg, is_train=False)
        elif dataset_name in ["refcocog_val_umd"]:
            mapper = RefCOCODatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name=dataset_name, mapper=mapper)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Modify train loader to use a fixed subset of dataset.
        """
        # Choose the appropriate dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "open_world_instance":
            mapper = OpenWorldSAM2InstanceDatasetMapper(cfg, is_train=True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "open_world_instance_all":
            mapper = OpenWorldSAM2InstanceDatasetMapperAll(cfg, is_train=True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "open_world_panoptic":
            mapper = OpenWorldSAM2PanopticDatasetMapper(cfg, is_train=True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "open_world_panoptic_all":
            mapper = OpenWorldSAM2PanopticDatasetMapperAll(cfg, is_train=True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "refcoco":
            mapper = RefCOCODatasetMapper(cfg, is_train=True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)


    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                        "relative_position_bias_table" in module_param_name
                        or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

def setup(args):
    """
    Create configs and perform basic setups.
    """

    cfg = get_cfg()
    cfg.set_new_allowed(True)  # Add this line before merging the file
    add_open_world_sam2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, f"run_{args.run_idx}")
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="open-world-sam2")
    return cfg

def set_seed(seed=42):
    # Set random seeds for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    set_seed()
    cfg = setup(args)
    # print("Command cfg:", cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        model.metadata = MetadataCatalog.get(cfg['DATASETS']['TEST'][0])
        print(cfg.OUTPUT_DIR)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--run_idx', default=0, type=int, metavar='N',
                        help='index of the experiment')
    parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', default=0.0001, type=float,)
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


