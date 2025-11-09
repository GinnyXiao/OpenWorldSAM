from typing import List, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import defaultdict

# modeing
from transformers import AutoTokenizer
from .evf_sam2 import EvfSam2Model
from .criterion import SetCriterion
from .matcher import HungarianMatcher
from .segment_anything_2.sam2.modeling.sam2_utils import MLP

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog
import logging


@META_ARCH_REGISTRY.register()
class OpenWorldSAM2(nn.Module):
    @configurable
    def __init__(
            self,
            *,
            evf_sam2: EvfSam2Model,
            tokenizer: AutoTokenizer,
            visual_model: nn.Module,
            mm_extractor: nn.Module,
            text_hidden_fcs: nn.ModuleList,
            query_dim: int,
            num_tokens: int,
            positional_tokens: nn.Parameter,
            criterion: nn.Module,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            dtype: torch.dtype,
            test_topk_per_image: int,
            top_k_on: bool,
            nms_on: bool,
            nms_threshold: float,
            iou_threshold: float,
            semantic_on: bool,
            instance_on: bool,
            panoptic_on: bool,
            sam_iou: bool,
            use_visual_tokens: bool = True,
            use_cross_attention: bool = False,
            cross_attention_layers: int = 3,  # Added parameter for number of layers
            two_stage_inference: bool = False,  # Add new parameter here
            refer_on: bool = False,  # Add refer_on parameter
            metadata: MetadataCatalog = None,
    ):
        super(OpenWorldSAM2, self).__init__()
        self.evf_sam2 = evf_sam2
        self.tokenizer = tokenizer
        self.visual_model = visual_model
        self.mm_extractor = mm_extractor
        self.text_hidden_fcs = text_hidden_fcs
        self.query_dim = query_dim  # query embedding dimension
        self.num_tokens = num_tokens
        self.criterion = criterion
        self.positional_tokens = positional_tokens
        self.use_visual_tokens = use_visual_tokens
        self.use_cross_attention = use_cross_attention
        self.metadata = metadata
        self.two_stage_inference = two_stage_inference  # Store the new parameter
        self.refer_on = refer_on  # Store refer_on parameter

        # Add cross-attention transformer if enabled
        if self.use_cross_attention:
            self.cross_attention_transformer = CrossAttentionTransformer(
                embedding_dim=256,
                num_heads=8,
                mlp_dim=query_dim * 4,
                num_layers=cross_attention_layers,  # Use the new parameter
                dropout=0.1
            )

        if not sam_iou:
            self.objectness_prediction_head = MLP(
                input_dim=evf_sam2.visual_model.memory_attention.d_model,
                hidden_dim=256,
                output_dim=1,
                num_layers=3,
                sigmoid_output=True,
            )
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.dtype = dtype

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.sam_iou = sam_iou
        self.top_k_on = top_k_on
        self.nms_on = nms_on
        self.test_topk_per_image = test_topk_per_image
        self.nms_threshold = nms_threshold
        self.iou_threshold = iou_threshold
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    @classmethod
    def from_config(cls, cfg):
        # EVF-SAM config & model
        evf_config = cfg.MODEL.OpenWorldSAM2.EVF_CONFIG
        torch_dtype = torch.float32
        kwargs = {"torch_dtype": torch_dtype,
                  "vision_pretrained": cfg.MODEL.OpenWorldSAM2.VISION_PRETRAINED,
                  "encoder_pretrained": cfg.MODEL.OpenWorldSAM2.ENCODER_PRETRAINED,
                  }

        # tokenizer
        tokenizer_config = cfg.MODEL.OpenWorldSAM2.TOKENIZER_CONFIG
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config, padding_side="right", use_fast=False)

        # EVF-SAM2 model
        evf_sam2 = EvfSam2Model.from_pretrained(evf_config, low_cpu_mem_usage=True, **kwargs)
        evf_sam2.config.eos_token_id = tokenizer.eos_token_id
        evf_sam2.config.bos_token_id = tokenizer.bos_token_id
        evf_sam2.config.pad_token_id = tokenizer.pad_token_id

        # SAM2 visual model
        visual_model = evf_sam2.visual_model
        for param in visual_model.parameters():
            param.requires_grad = False

        # BEiT-3 model
        mm_extractor = evf_sam2.mm_extractor
        if cfg.MODEL.OpenWorldSAM2.TRAIN_VLM:
            for param in mm_extractor.parameters():
                param.requires_grad = True
        else:
            for param in mm_extractor.parameters():
                param.requires_grad = False

        # Projection Layer
        query_dim = cfg.MODEL.OpenWorldSAM2.QUERY_DIM
        in_dim = evf_sam2.config.hidden_size
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, query_dim)
        ]
        text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        text_hidden_fcs.train()
        for param in text_hidden_fcs.parameters():
            param.requires_grad = True

        # OpenWorldSAM2 config
        num_tokens = cfg.MODEL.OpenWorldSAM2.NUM_OBJECT_QUERIES
        positional_tokens = nn.Parameter(torch.randn(num_tokens, query_dim))
        positional_tokens.requires_grad = True

        # Loss parameters:
        no_object_weight = cfg.MODEL.OpenWorldSAM2.NO_OBJECT_WEIGHT
        dice_weight = cfg.MODEL.OpenWorldSAM2.DICE_WEIGHT
        mask_weight = cfg.MODEL.OpenWorldSAM2.MASK_WEIGHT
        objectness_weight = cfg.MODEL.OpenWorldSAM2.OBJECTNESS_WEIGHT

        # Get use_cross_attention from config
        use_cross_attention = getattr(cfg.MODEL.OpenWorldSAM2, "USE_CROSS_ATTENTION", False)
        # Get two_stage_inference from config with default=False
        two_stage_inference = getattr(cfg.MODEL.OpenWorldSAM2.TEST, "TWO_STAGE_INFERENCE", False)
        # Get refer_on from config with default=False
        refer_on = getattr(cfg.MODEL.OpenWorldSAM2.TEST, "REFER_ON", False)

        # building criterion
        matcher = HungarianMatcher(
            cost_class=objectness_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
        )

        weight_dict = {"loss_ce": objectness_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            num_classes=1,  # omitting the special no-object category, 1 class to indicate object or no object
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
        )

        return {
            "evf_sam2": evf_sam2,
            "tokenizer": tokenizer,
            "visual_model": visual_model,
            "mm_extractor": mm_extractor,
            "text_hidden_fcs": text_hidden_fcs,
            "query_dim": query_dim,
            "num_tokens": num_tokens,
            "positional_tokens": positional_tokens,
            "criterion": criterion,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "dtype": torch_dtype,
            # inference
            "semantic_on": cfg.MODEL.OpenWorldSAM2.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.OpenWorldSAM2.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.OpenWorldSAM2.TEST.PANOPTIC_ON,
            "top_k_on": cfg.MODEL.OpenWorldSAM2.TEST.TOP_K_ON,
            "nms_on": cfg.MODEL.OpenWorldSAM2.TEST.NMS_ON,
            "test_topk_per_image": cfg.MODEL.OpenWorldSAM2.TEST.DETECTIONS_PER_IMAGE,
            "nms_threshold": cfg.MODEL.OpenWorldSAM2.TEST.NMS_THRESHOLD,
            "iou_threshold": cfg.MODEL.OpenWorldSAM2.TEST.IOU_THRESHOLD,
            "sam_iou": cfg.MODEL.OpenWorldSAM2.SAM_IOU,
            "use_visual_tokens": cfg.MODEL.OpenWorldSAM2.USE_VISUAL_TOKENS,
            "use_cross_attention": use_cross_attention,
            "cross_attention_layers": cfg.MODEL.OpenWorldSAM2.CROSS_ATTENTION_LAYERS,
            "two_stage_inference": two_stage_inference,  # Add the new parameter here
            "refer_on": refer_on,  # Add refer_on from config
            "metadata": MetadataCatalog.get(cfg['DATASETS']['TRAIN'][0]) ,
        }

    def print_trainable_parameters(self):
        """
        Prints the names and number of trainable parameters in the model.
        """
        logger = logging.getLogger("detectron2")

        total_params = 0
        trainable_params = 0
        logger.info(f"{'Parameter Name':<40}{'Trainable':<10}{'Shape':<20}{'Num Params':<15}")
        logger.info("=" * 85)

        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
                trainable_status = "Yes"
                logger.info(f"{name:<40}{trainable_status:<10}{str(list(param.shape)):<20}{num_params:<15}")
            else:
                trainable_status = "No"
            #

        logger.info("=" * 85)
        logger.info(f"Total parameters: {total_params}")
        logger.info(f"Trainable parameters: {trainable_params}")
        logger.info(f"Non-trainable parameters: {total_params - trainable_params}")
        logger.info("=" * 85)
        logger.info(f"use_cross_attention: {self.use_cross_attention}")
        logger.info(f"use_visual_tokens: {self.use_visual_tokens}")
        logger.info(f"two_stage_inference: {self.two_stage_inference}")

    @property
    def device(self):
        return self.pixel_mean.device

    def tokenize_prompts(self, prompts: List):
        input_ids = [
            self.tokenizer(prompt, return_tensors="pt").input_ids[0]
            for prompt in prompts
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id)

        truncate_len = self.tokenizer.model_max_length

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]
        return input_ids.to(self.device), attention_masks.to(self.device)

    def forward(
            self,
            batched_inputs,
            return_intermediate=False
    ):
        """
                Args:
                    batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                        Each item in the list contains the inputs for one image.
                        For now, each item in the list is a dict that contains:
                           * "image": Tensor, image in (C, H, W) format.
                           * "instances": per-region ground truth
                           * Other information that's included in the original dicts, such as:
                             "height", "width" (int): the output resolution of the model (may be different
                             from input resolution), used in inference.
                           * prompt: String, a prompt for the corresponding image
                           * input_id: Tensor, the tokenized input_ids for the corresponding prompt
                Returns:
                    dict[str, Tensor]:
                """

        ######################## input pre-processing #######################
        images = [x["image"].to(dtype=self.dtype, device=self.device) for x in batched_inputs]
        original_size_list = [(x["height"], x["width"]) for x in batched_inputs]
        images_evf = [x["evf_image"].to(dtype=self.dtype, device=self.device) for x in batched_inputs]
        # Convert to tensors
        images = ImageList.from_tensors(images, 1024).tensor
        images_evf = ImageList.from_tensors(images_evf, 224).tensor

        # Calculate offsets for prompts per image
        offset = [0]
        all_prompts = []

        # Process each image and build prompts list
        for x in batched_inputs:
            prompts = x["prompt"]
            all_prompts.extend(prompts)
            offset.append(offset[-1] + len(prompts))

        input_ids, attention_masks = self.tokenize_prompts(all_prompts)
        batch_size = len(batched_inputs)
        assert batch_size == len(offset) - 1

        ############################## forward #############################
        backbone_out = self.visual_model.forward_image(images)
        # dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn'])

        _, image_embeddings, _, _ = self.visual_model._prepare_backbone_features(backbone_out)

        # Expand images_evf according to number of prompts per image
        if self.use_visual_tokens:
            images_evf_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_evf_i = (
                    images_evf[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_evf_list.append(images_evf_i)
            images_evf = torch.cat(images_evf_list, dim=0)

            # Process through BEIT-3
            output = self.mm_extractor.beit3(
                visual_tokens=images_evf,
                textual_tokens=input_ids,
                text_padding_position=~attention_masks,
            )
        else:
            # When not using visual tokens, we'll pass None
            output = self.mm_extractor.beit3(
                visual_tokens=None,
                textual_tokens=input_ids,
                text_padding_position=~attention_masks,
            )
        feat = output["encoder_out"][:, :1, ...]
        feat = self.text_hidden_fcs[0](feat)

        # Split features back according to images
        """
        Within a single image of the training dataset, there are several (usually more than 1)
        referring expressions corresponding to different parts of the image. For example we
        use batch 2 to train the code, and the first image has 3 referring expressions and the
        secode image has 2 referring expressions, the offset would be [0, 3, 5]. The torch.split
        would split the multi-modal extracted feat of length 5 to a list, where each item of the
        list corresponds to each image in batch.
        """
        feat = torch.split(feat, [offset[i + 1] - offset[i] for i in range(len(offset) - 1)])
        # print(f"Split features length: {len(feat)}, First feature shape: {feat[0].shape}")

        # Process image features
        image_embeddings = [_.to(images.dtype) for _ in image_embeddings]
        if self.visual_model.directly_add_no_mem_embed:
            image_embeddings[-1] = image_embeddings[-1] + self.visual_model.no_mem_embed

        feats = [
                    feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                    for feat, feat_size in zip(image_embeddings[::-1], self._bb_feat_sizes[::-1])
                ][::-1]

        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        if self.training:
            # Initialize lists to store all predictions and losses
            all_losses = defaultdict(list)

        if not self.training:
            processed_results = []

        # Process each image batch
        for img_idx in range(batch_size):
            img_feat = feat[img_idx]  # Get features for all prompts of this image

            # Prepare all feat_with_tokens for this image's prompts
            batch_feat_with_tokens = []
            for prompt_idx, prompt_feat in enumerate(img_feat):
                # Repeat feature along token dimension and add positional embeddings
                feat_repeated = prompt_feat.expand(self.num_tokens, -1, -1)
                feat_with_tokens = feat_repeated + self.positional_tokens.unsqueeze(1)
                batch_feat_with_tokens.append(feat_with_tokens)

            # Concatenate all prompts for this image
            batch_feat_with_tokens = torch.cat(batch_feat_with_tokens, dim=0)

            # Apply cross-attention if enabled
            if self.use_cross_attention:
                # Prepare image embeddings for cross-attention
                img_embed = _features["image_embed"][img_idx]  # [C, H, W]
                img_embed = img_embed.flatten(1).transpose(0, 1)  # [H*W, C]

                # Add a batch dimension to img_embed to make it 3D [1, H*W, C]
                img_embed = img_embed.unsqueeze(0)

                # Apply cross-attention
                original_batch_feat_with_tokens = batch_feat_with_tokens

                # Reshape batch_feat_with_tokens to be 3D [batch_size, num_tokens, embedding_dim]
                # The current shape is likely [batch_size, num_tokens, 1, embedding_dim]
                if batch_feat_with_tokens.dim() == 3:
                    reshaped_batch_feat = batch_feat_with_tokens.squeeze(1)
                else:
                    reshaped_batch_feat = batch_feat_with_tokens

                enhanced_batch_feat_with_tokens = self.cross_attention_transformer(
                    reshaped_batch_feat.unsqueeze(0),  # Add batch dimension [1, num_tokens, embedding_dim]
                    img_embed
                )
                # Remove batch dimension
                enhanced_batch_feat_with_tokens = enhanced_batch_feat_with_tokens.squeeze(0)

                # Reshape back to original shape if needed
                if batch_feat_with_tokens.dim() == 2:
                    enhanced_batch_feat_with_tokens = enhanced_batch_feat_with_tokens.unsqueeze(1)

                # Skip connection
                batch_feat_with_tokens = original_batch_feat_with_tokens + enhanced_batch_feat_with_tokens

            # print(f"Batch feat with tokens shape: {batch_feat_with_tokens.shape}")

            # Process all prompts for this image through SAM prompt encoder
            sparse_embeddings, dense_embeddings = self.visual_model.sam_prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=batch_feat_with_tokens,
            )
            sparse_embeddings = sparse_embeddings.to(batch_feat_with_tokens.dtype)

            high_res_features = [
                feat_level[img_idx].unsqueeze(0)
                for feat_level in _features["high_res_feats"]
            ]

            # Process all prompts for this image through SAM mask decoder
            low_res_masks, iou_pred, sam_tokens_out, _ = self.visual_model.sam_mask_decoder(
                image_embeddings=_features["image_embed"][img_idx].unsqueeze(0),
                image_pe=self.visual_model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=True,
                high_res_features=high_res_features,
            )

            # Get predictions for this image
            pred_masks = low_res_masks.squeeze(1)
            if not self.sam_iou:
                objectness_logits = self.objectness_prediction_head(sam_tokens_out.squeeze(1))
                outputs = {"pred_masks": pred_masks.unsqueeze(0), "pred_logits": objectness_logits.unsqueeze(0)}
            else:
                outputs = {"pred_masks": pred_masks.unsqueeze(0), "pred_logits": iou_pred.unsqueeze(0)}

            ################################# Inference Postprocessing ##################################
            # Postprocess masks
            if not self.training:
                unique_categories = batched_inputs[img_idx]["unique_categories"]

                # Assign class labels before filtering
                num_total_masks = len(pred_masks)
                # Each unique category gets num_tokens number of predictions
                class_indices = torch.div(torch.arange(num_total_masks, device=self.device),
                                          self.num_tokens, rounding_mode='floor')
                # Map to actual category IDs from unique_categories
                class_labels = torch.tensor([unique_categories[i] for i in class_indices],
                                            dtype=torch.int64,
                                            device=self.device)

                # FIRST STAGE FILTERING: Filter out low IoU predictions before second stage
                pred_logits = outputs["pred_logits"].squeeze(0)

                iou_scores = pred_logits.squeeze(1) if pred_logits.dim() > 1 else pred_logits
                
                # Only apply two-stage inference if enabled
                if self.two_stage_inference:
                    # Apply IoU threshold to filter masks
                    keep_indices = iou_scores >= self.iou_threshold
                    
                    if keep_indices.sum() > 0:
                        # Filter masks based on IoU scores
                        filtered_masks = low_res_masks[keep_indices]
                        filtered_class_labels = class_labels[keep_indices]
                    
                        # SECOND STAGE: Use filtered masks as visual prompts for SAM
                        sparse_embeddings, dense_embeddings = self.visual_model.sam_prompt_encoder(
                            points=None,
                            boxes=None,
                            masks=filtered_masks,
                            text_embeds=None,
                        )
                    
                        refined_masks, refined_iou_pred, refined_tokens_out, _ = self.visual_model.sam_mask_decoder(
                            image_embeddings=_features["image_embed"][img_idx].unsqueeze(0),
                            image_pe=self.visual_model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                            repeat_image=True,
                            high_res_features=high_res_features,
                        )
                    
                        # Update low_res_masks and outputs with refined predictions
                        low_res_masks = refined_masks
                        pred_logits = refined_iou_pred
                        class_labels = filtered_class_labels

                # Proceed with postprocessing using the refined masks
                pred_masks = self.postprocess_masks(low_res_masks, orig_hw=original_size_list[img_idx])

                processed_results.append({})

                if self.refer_on:
                    # Get referring expression masks
                    refer_masks, refer_scores  = self.refer_inference(pred_masks, pred_logits, class_labels)
                    processed_results[-1]["grounding_mask"] = refer_masks
                    processed_results[-1]["grounding_scores"] = refer_scores

                if self.instance_on:
                    # Process all predictions and perform NMS
                    prompt_results = self.instance_inference(pred_masks, pred_logits, class_labels)

                    # Add instance segmentation results
                    processed_results[-1]["instances"] = prompt_results

                if self.panoptic_on:
                    # Generate panoptic segmentation directly from predictions
                    # No need to rely on instance results
                    panoptic_r = self.panoptic_inference(
                        pred_logits,  # [num_queries, 1]
                        pred_masks,  # [num_queries, 1, H, W]
                        class_labels  # [num_queries]
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                if self.semantic_on:
                    # Prepare inputs for semantic inference
                    # Create one-hot class scores
                    num_classes = len(self.metadata.stuff_classes)
                    mask_cls = torch.zeros((pred_masks.shape[0], num_classes + 1),
                                           device=self.device)  # +1 for background

                    # Fill in class scores based on class labels and prediction scores
                    for idx, (cls_id, score) in enumerate(zip(class_labels, pred_logits.squeeze(1))):
                        mask_cls[idx, cls_id] = score

                    # Generate semantic segmentation
                    sem_seg = self.semantic_inference(mask_cls, pred_masks, keep_sem_bgd=False)
                    processed_results[-1]["sem_seg"] = sem_seg

                return processed_results

            ################################# Calculate Losses #######################################
            # Calculate loss for this image if in training mode
            if self.training:
                gt_instances = batched_inputs[img_idx]["instances"]
                if not isinstance(gt_instances, list):
                    gt_instances = [gt_instances]

                # For per-prompt matching, we need to split the predictions by prompt
                num_prompts = len(gt_instances)

                # Each prompt gets self.num_tokens predictions
                pred_splits = [self.num_tokens] * num_prompts
                pred_masks_list = torch.split(pred_masks, pred_splits)

                if not self.sam_iou:
                    pred_logits_list = torch.split(objectness_logits, pred_splits)
                else:
                    pred_logits_list = torch.split(iou_pred, pred_splits)

                # Process each prompt separately
                for prompt_idx in range(num_prompts):
                    # Create outputs for this prompt
                    prompt_outputs = {
                        "pred_masks": pred_masks_list[prompt_idx].unsqueeze(0),
                        "pred_logits": pred_logits_list[prompt_idx].unsqueeze(0)
                    }

                    # Prepare targets for this prompt
                    prompt_targets = self.prepare_targets([gt_instances[prompt_idx]])

                    if return_intermediate and prompt_idx == 0:
                        return prompt_outputs, prompt_targets

                    # Calculate losses for this prompt
                    prompt_losses = self.criterion(prompt_outputs, prompt_targets)

                    # Store weighted losses
                    for k, v in prompt_losses.items():
                        if k in self.criterion.weight_dict:
                            all_losses[k].append(v * self.criterion.weight_dict[k])

        # Average losses across batch
        if self.training:
            final_losses = {k: torch.stack(v).mean() for k, v in all_losses.items()}
            return final_losses

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks.to(dtype=self.dtype, device=self.device)
            # unlike traditional instance segmentation model that predicts for every instance,
            # we only want instances that correspond to the prompt queries (conditional predictions),
            # so we set the labels to 0 for all instances (label doesn't matter for conditional predictions)
            labels = torch.zeros_like(targets_per_image.gt_classes).to(device=self.device)

            target_dict = {
                "labels": labels,
                "masks": gt_masks,
            }

            new_targets.append(target_dict)
        return new_targets

    def instance_inference(self, pred_masks, iou_scores, class_labels):
        """
        Postprocess predicted masks and IoU scores to generate instance segmentation results.

        Args:
            pred_masks (Tensor): Predicted masks of shape [num_queries, H, W].
            iou_scores (Tensor): IoU scores of shape [num_queries, 1].
            class_labels (Tensor): Class labels of shape [num_queries].

        Returns:
            Instances: An `Instances` object containing the final masks, boxes, scores, and class IDs.
        """
        test_topk_per_image = self.test_topk_per_image
        nms_threshold = self.nms_threshold
        iou_threshold = self.iou_threshold  # Filtering IoU threshold
        top_k = self.top_k_on
        nms = self.nms_on

        image_size = pred_masks.shape[-2:]
        iou_scores = iou_scores.squeeze(1)  # Shape: [num_queries]
        pred_masks = pred_masks.squeeze(1)  # Shape: [num_queries, H, W]

        if self.panoptic_on:
            thing_dataset_id_to_contiguous_id = self.metadata.thing_dataset_id_to_contiguous_id if hasattr(
                self.metadata, 'thing_dataset_id_to_contiguous_id') else {}
            keep = torch.zeros_like(iou_scores).bool()
            for i, lab in enumerate(class_labels):
                keep[i] = lab in thing_dataset_id_to_contiguous_id.values()

            pred_masks = pred_masks[keep]
            iou_scores = iou_scores[keep]
            class_labels = class_labels[keep]

        # Step 1: Select top-k masks based on IoU scores
        if top_k:
            top_k = min(test_topk_per_image, pred_masks.shape[0])  # Ensure top_k does not exceed the number of masks
            top_k_indices = torch.argsort(iou_scores, descending=True)[:top_k]

            pred_masks = pred_masks[top_k_indices]
            iou_scores = iou_scores[top_k_indices]
            class_labels = class_labels[top_k_indices]

        # Step 2: Filter masks based on IoU threshold
        keep_indices = iou_scores >= iou_threshold
        pred_masks = pred_masks[keep_indices]
        iou_scores = iou_scores[keep_indices]
        class_labels = class_labels[keep_indices]

        if pred_masks.shape[0] == 0:
            # No valid masks remain after filtering
            print("No valid masks remain after filtering. Returning an empty Instances object.")
            # Return an empty Instances object
            result = Instances(image_size)
            result.pred_masks = torch.empty((0, image_size[0], image_size[1]), device=self.device)
            result.pred_boxes = Boxes(torch.empty((0, 4), device=self.device))
            result.scores = torch.empty((0,), device=self.device)
            result.pred_classes = torch.empty((0,), dtype=torch.int64, device=self.device)
            return result

        # Step 3: Compute bounding boxes from masks
        bit_masks = BitMasks(pred_masks > 0)  # Binarize masks
        pred_boxes = bit_masks.get_bounding_boxes().to(device=self.device)  # Shape: [num_instances, 4]

        # Step 4: Non-Maximum Suppression (NMS)
        if nms:
            nms_keep = torchvision.ops.nms(pred_boxes.tensor, iou_scores, nms_threshold)
            pred_masks = pred_masks[nms_keep]
            pred_boxes = pred_boxes[nms_keep]
            iou_scores = iou_scores[nms_keep]
            class_labels = class_labels[nms_keep]

        # Step 5: Create Instances
        result = Instances(image_size)
        result.pred_masks = (pred_masks > 0).float()
        result.pred_boxes = pred_boxes
        result.scores = iou_scores
        result.pred_classes = class_labels
        return result

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        masks = masks.float()
        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks

    def semantic_inference(self, mask_cls, mask_pred, keep_sem_bgd=False):
        """
        Compute semantic segmentation predictions from class scores and predicted masks.

        Args:
            mask_cls (Tensor): Class logits of shape [num_queries, num_classes].
            mask_pred (Tensor): Binary mask logits of shape [num_queries, H, W].
            keep_sem_bgd (bool): Whether to keep background class or not.

        Returns:
            Tensor: Semantic segmentation of shape [num_classes, H, W].
        """

        if keep_sem_bgd:
            mask_cls = F.softmax(mask_cls, dim=-1)
        else:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]  # Remove background class

            # mask_pred = mask_pred.sigmoid()
        mask_pred = mask_pred.sigmoid()
        mask_pred = mask_pred.squeeze(1)
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def mask_nms(self, masks, scores, iou_threshold=0.5):
        """
        Apply Non-Maximum Suppression to masks based on their IoU and scores.

        Args:
            masks (Tensor): Binary masks of shape [N, H, W]
            scores (Tensor): Confidence scores of shape [N]
            iou_threshold (float): IoU threshold for suppression

        Returns:
            Tensor: Boolean tensor of shape [N] indicating which masks to keep
        """
        n = masks.shape[0]
        if n == 0:
            return torch.zeros(0, dtype=torch.bool, device=masks.device)
        if n == 1:
            return torch.ones(1, dtype=torch.bool, device=masks.device)

        # Ensure masks are binary
        binary_masks = masks >= 0.5

        # Calculate areas of each mask
        areas = binary_masks.sum(dim=(1, 2))

        # Sort by score
        order = torch.argsort(scores, descending=True)

        keep = torch.ones(n, dtype=torch.bool, device=masks.device)

        for i in range(n):
            # Skip if this mask is already suppressed
            if not keep[order[i]]:
                continue

            # Get the current mask
            mask_i = binary_masks[order[i]]
            area_i = areas[order[i]]

            # Check against all lower-scored masks
            for j in range(i + 1, n):
                if not keep[order[j]]:
                    continue

                # Calculate IoU
                mask_j = binary_masks[order[j]]
                area_j = areas[order[j]]

                intersection = (mask_i & mask_j).sum()
                union = area_i + area_j - intersection
                iou = intersection / union if union > 0 else 0

                # Suppress mask_j if IoU is above threshold
                if iou > iou_threshold:
                    keep[order[j]] = False

        return keep

    def panoptic_inference(self, mask_cls, mask_pred, class_labels):
        """
        Compute panoptic segmentation predictions from class scores and predicted masks.

        Args:
            mask_cls (Tensor): Class confidence scores of shape [num_queries, 1]
            mask_pred (Tensor): Binary masks of shape [num_queries, H, W]
            class_labels (Tensor): Class labels of shape [num_queries]

        Returns:
            Tuple: (panoptic_seg, segments_info)
                - panoptic_seg (Tensor): Panoptic segmentation of shape [H, W]
                - segments_info (List[Dict]): List of dictionaries containing information about each segment
        """
        scores = mask_cls.squeeze(1)  # [num_queries]
        mask_pred = mask_pred.squeeze(1)
        mask_pred = mask_pred.sigmoid()  # [num_queries, H, W]

        # Filter based on score threshold
        keep = scores > self.iou_threshold

        cur_scores = scores[keep]
        cur_classes = class_labels[keep]
        cur_masks = mask_pred[keep]

        # Get image dimensions
        h, w = cur_masks.shape[-2:]

        # Initialize panoptic segmentation tensor
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=self.device)
        segments_info = []

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask
            return panoptic_seg, segments_info

        # Apply NMS per class to remove duplicate predictions
        class_ids = torch.unique(cur_classes)
        nms_keep = torch.zeros_like(cur_scores, dtype=torch.bool)

        for cls_id in class_ids:
            # Find all masks for this class
            cls_mask = cur_classes == cls_id
            if cls_mask.sum() <= 1:
                # If only one mask for this class, keep it
                nms_keep[cls_mask] = True
                continue

            # Apply NMS to masks of this class
            cls_keep = self.mask_nms(
                cur_masks[cls_mask],
                cur_scores[cls_mask],
                iou_threshold=self.nms_threshold  # NMS IoU threshold
            )

            # Update the overall keep mask
            nms_keep[torch.where(cls_mask)[0][cls_keep]] = True

        # Apply NMS filtering
        cur_scores = cur_scores[nms_keep]
        cur_classes = cur_classes[nms_keep]
        cur_masks = cur_masks[nms_keep]

        # Calculate probabilities for each mask
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        # Take argmax to determine which mask has highest probability at each pixel
        cur_mask_ids = cur_prob_masks.argmax(0)

        # Track stuff (non-thing) regions to merge them
        stuff_memory_list = {}

        # Get information about which classes are "things" vs. "stuff"
        thing_dataset_id_to_contiguous_id = {}
        if hasattr(self.metadata, 'thing_dataset_id_to_contiguous_id'):
            thing_dataset_id_to_contiguous_id = self.metadata.thing_dataset_id_to_contiguous_id

        # Process each mask
        current_segment_id = 0
        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()
            isthing = pred_class in thing_dataset_id_to_contiguous_id.values()

            # Get mask area statistics
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

            # Skip masks with small valid areas or overlap issues
            # Use a more relaxed threshold since we've already handled duplicates with NMS
            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < 0.5:  # Relaxed from 0.8 to 0.5
                    continue

                # Merge stuff regions with same class
                if not isthing:
                    if int(pred_class) in stuff_memory_list.keys():
                        panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                        continue
                    else:
                        stuff_memory_list[int(pred_class)] = current_segment_id + 1

                # Update panoptic segmentation
                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id

                # Add segment info
                seg_info = {
                    "id": current_segment_id,
                    "isthing": bool(isthing),
                    "category_id": int(pred_class),
                }
                segments_info.append(seg_info)

        return panoptic_seg, segments_info


    def refer_inference(self, pred_masks, pred_logits, class_labels):
        """
        For each class, identify the mask prediction that has the highest confidence score.
        
        Args:
            pred_masks (Tensor): Predicted masks of shape [num_queries, H, W]
            pred_logits (Tensor): Confidence scores of shape [num_queries, 1]
            class_labels (Tensor): Class labels of shape [num_queries]
            
        Returns:
            Tensor: Mask predictions of shape [num_classes, H, W]
        """
        # Get unique class labels
        unique_classes = torch.unique(class_labels)
        num_classes = len(unique_classes)
        h, w = pred_masks.shape[-2:]
        
        # Initialize output tensor
        class_masks = torch.zeros((num_classes, h, w), device=self.device)
        class_scores = torch.zeros((num_classes), device=self.device)
        # For each class, find the mask with highest confidence
        for i, cls in enumerate(unique_classes):
            # Get indices for this class
            cls_indices = (class_labels == cls)
            
            if cls_indices.sum() > 0:
                # Get masks and scores for this class
                cls_masks = pred_masks[cls_indices]
                cls_scores = pred_logits[cls_indices].squeeze(-1)
                
                # Find mask with highest score
                best_idx = torch.argmax(cls_scores)
                best_mask = cls_masks[best_idx]
                best_score = cls_scores[best_idx]
                # Store in output tensor
                class_masks[i] = best_mask
                class_scores[i] = best_score
                
        return class_masks, class_scores


# Add the CrossAttentionTransformer class after the OpenWorldSAM2 class definition
class CrossAttentionTransformer(nn.Module):
    """
    A stack of Transformer blocks for cross-attention between VLM features and image embeddings.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int,
            num_layers: int = 3,  # Added parameter for number of layers
            dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Create a stack of transformer layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Add projection layers to handle dimension mismatches
        self.input_projection = None
        self.image_projection = None

    def forward(
            self,
            vlm_features: torch.Tensor,  # [batch_size, num_tokens, embedding_dim]
            image_embeddings: torch.Tensor,  # [batch_size, H*W, embedding_dim]
    ) -> torch.Tensor:
        """
        Forward pass through multiple layers of cross-attention.

        Args:
            vlm_features: Tensor of shape [batch_size, num_tokens, embedding_dim]
            image_embeddings: Tensor of shape [batch_size, H*W, embedding_dim]

        Returns:
            Tensor of shape [batch_size, num_tokens, embedding_dim]
        """
        # Ensure inputs are 3D tensors with batch dimension
        assert vlm_features.dim() == 3, f"vlm_features should be 3D, got shape {vlm_features.shape}"
        assert image_embeddings.dim() == 3, f"image_embeddings should be 3D, got shape {image_embeddings.shape}"

        # Check if we need to create projection layers for dimension mismatch
        input_dim = vlm_features.size(-1)
        image_dim = image_embeddings.size(-1)

        # Create projection layers if needed and if they don't exist yet
        if input_dim != self.embedding_dim and self.input_projection is None:
            print(f"Creating input projection layer from {input_dim} to {self.embedding_dim}")
            self.input_projection = nn.Linear(input_dim, self.embedding_dim).to(vlm_features.device)

        if image_dim != self.embedding_dim and self.image_projection is None:
            print(f"Creating image projection layer from {image_dim} to {self.embedding_dim}")
            self.image_projection = nn.Linear(image_dim, self.embedding_dim).to(image_embeddings.device)

        # Apply projections if needed
        if self.input_projection is not None:
            vlm_features = self.input_projection(vlm_features)

        if self.image_projection is not None:
            image_embeddings = self.image_projection(image_embeddings)

        # Pass through all layers
        x = vlm_features
        for layer in self.layers:
            x = layer(x, image_embeddings)

        # Project back to original dimension if needed
        if self.input_projection is not None:
            # Create a projection back to the original dimension
            if not hasattr(self, 'output_projection') or self.output_projection is None:
                print(f"Creating output projection layer from {self.embedding_dim} to {input_dim}")
                self.output_projection = nn.Linear(self.embedding_dim, input_dim).to(x.device)
            x = self.output_projection(x)

        return x


class CrossAttentionLayer(nn.Module):
    """
    A single Transformer layer for cross-attention between VLM features and image embeddings.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Self-attention for VLM features
        self.self_attn_norm = nn.LayerNorm(embedding_dim)
        self.self_attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_dropout = nn.Dropout(dropout)

        # Cross-attention from VLM features to image embeddings
        self.cross_attn_norm = nn.LayerNorm(embedding_dim)
        self.cross_attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_dropout = nn.Dropout(dropout)

        # MLP block
        self.mlp_norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(
            self,
            vlm_features: torch.Tensor,  # [batch_size, num_tokens, embedding_dim]
            image_embeddings: torch.Tensor,  # [batch_size, H*W, embedding_dim]
    ) -> torch.Tensor:
        """
        Forward pass for a single cross-attention layer.

        Args:
            vlm_features: Tensor of shape [batch_size, num_tokens, embedding_dim]
            image_embeddings: Tensor of shape [batch_size, H*W, embedding_dim]

        Returns:
            Tensor of shape [batch_size, num_tokens, embedding_dim]
        """
        # Self-attention
        residual = vlm_features
        x = self.self_attn_norm(vlm_features)
        x, _ = self.self_attn(x, x, x)
        x = self.self_attn_dropout(x)
        x = residual + x

        # Cross-attention
        residual = x
        x = self.cross_attn_norm(x)
        x, _ = self.cross_attn(
            query=x,
            key=image_embeddings,
            value=image_embeddings
        )
        x = self.cross_attn_dropout(x)
        x = residual + x

        # MLP
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x

        return x