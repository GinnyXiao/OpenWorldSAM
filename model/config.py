from detectron2.config import CfgNode as CN

def add_open_world_sam2_config(cfg):
    """
    Add config for OpenWorldSAM.
    """
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "open_world_instance"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # OWSAM model config
    cfg.MODEL.OpenWorldSAM2 = CN()

    # EVF-SAM model config
    cfg.MODEL.OpenWorldSAM2.EVF_CONFIG = "YxZhang/evf-sam2-multitask"
    cfg.MODEL.OpenWorldSAM2.TOKENIZER_CONFIG = "YxZhang/evf-sam2-multitask"
    cfg.MODEL.OpenWorldSAM2.TORCH_DTYPE = "fp32" # choices=["fp32", "bf16", "fp16"]
    cfg.MODEL.OpenWorldSAM2.TRAIN_MASK_DECODER = False
    cfg.MODEL.OpenWorldSAM2.TRAIN_PROMPT_ENCODER = False
    cfg.MODEL.OpenWorldSAM2.TRAIN_VLM = False
    cfg.MODEL.OpenWorldSAM2.QUERY_DIM = 256
    cfg.MODEL.OpenWorldSAM2.VISION_PRETRAINED = "checkpoints/sam_vit_h_4b8939.pth"

    # OPENWORLDSAM2 config
    cfg.MODEL.OpenWorldSAM2.NUM_OBJECT_QUERIES = 20
    cfg.MODEL.OpenWorldSAM2.TRAIN_TIE_BREAKER = True
    cfg.MODEL.OpenWorldSAM2.USE_VISUAL_TOKENS = True
    cfg.MODEL.OpenWorldSAM2.USE_CROSS_ATTENTION = True
    cfg.MODEL.OpenWorldSAM2.CROSS_ATTENTION_LAYERS = 3

    # OPENWORLDSAM2 inference config
    cfg.MODEL.OpenWorldSAM2.TEST = CN()
    cfg.MODEL.OpenWorldSAM2.TEST.SEMANTIC_ON = False
    cfg.MODEL.OpenWorldSAM2.TEST.INSTANCE_ON = True
    cfg.MODEL.OpenWorldSAM2.TEST.PANOPTIC_ON = False
    cfg.MODEL.OpenWorldSAM2.TEST.TOP_K_ON = False
    cfg.MODEL.OpenWorldSAM2.TEST.NMS_ON = True
    cfg.MODEL.OpenWorldSAM2.TEST.NMS_THRESHOLD = 0.0
    cfg.MODEL.OpenWorldSAM2.TEST.IOU_THRESHOLD = 0.0
    cfg.MODEL.OpenWorldSAM2.TEST.DETECTIONS_PER_IMAGE = 30
    cfg.MODEL.OpenWorldSAM2.TEST.TWO_STAGE_INFERENCE = False
    cfg.MODEL.OpenWorldSAM2.TEST.REFER_ON = False

    # loss
    cfg.MODEL.OpenWorldSAM2.DICE_WEIGHT = 1.0
    cfg.MODEL.OpenWorldSAM2.MASK_WEIGHT = 5.0
    # no object and objectness weight are set to 0 to disable them
    # no supervision on SAM's IOU prediction because empirical results show it does not help
    cfg.MODEL.OpenWorldSAM2.NO_OBJECT_WEIGHT = 0.0
    cfg.MODEL.OpenWorldSAM2.OBJECTNESS_WEIGHT = 0.0

