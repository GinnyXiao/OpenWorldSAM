# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size

from .utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(inputs, targets, num_masks, smooth= 1):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + smooth) / (denominator + smooth)
    return loss.sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        
        # Extract class_weight from weight_dict, default to 1.0 if not present
        self.class_weight = weight_dict.get("loss_classes", 1.0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]  # [bs, num_queries, 1]

        # Handle positive samples
        # Get indices for object predictions
        batch_idx, src_idx = self._get_src_permutation_idx(indices)
        object_logits = src_logits[batch_idx, src_idx].squeeze(-1)  # Shape: [num_objects]
        
        # Add numerical stability - clip values to prevent extreme values
        object_logits = torch.clamp(object_logits, min=-100.0, max=100.0)
        
        # Step 1: Calculate the object loss as (1 - src_logits[idx]) with safeguard
        if object_logits.numel() > 0:
            object_loss = (1 - object_logits).mean()
        else:
            object_loss = torch.tensor(0.0, device=src_logits.device)

        # Step 2: Create a mask for non-object indices
        mask = torch.ones_like(src_logits, dtype=torch.bool)
        mask[batch_idx, src_idx] = False  # Set object indices to False

        # Step 3: Calculate the non-object loss with `no_object_weight` and safeguards
        non_object_logits = src_logits[mask].squeeze(-1)  # Flatten to [num_non_objects]
        
        # Add numerical stability - clip values to prevent extreme values
        non_object_logits = torch.clamp(non_object_logits, min=-100.0, max=100.0)
        
        if non_object_logits.numel() > 0:
            non_object_loss = (non_object_logits * self.eos_coef).mean()
        else:
            non_object_loss = torch.tensor(0.0, device=src_logits.device)
        
        # Step 4: Sum the object and non-object losses with safeguards
        loss_ce = object_loss + non_object_loss
        
        # Extra safeguard against NaN
        if torch.isnan(loss_ce) or torch.isinf(loss_ce):
            print(f"Warning: NaN or Inf detected in loss_ce. Using zero loss instead.")
            loss_ce = torch.tensor(0.0, device=src_logits.device)

        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        # Continue with regular mask loss calculation
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = F.interpolate(
            src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_masks),
            "loss_dice": dice_loss(src_masks, target_masks, num_masks),
        }
        return losses

    def loss_classes(self, outputs, targets, indices, num_masks):
        """
        Compute the classification loss using focal loss for semantic class prediction.
        
        Args:
            outputs: Dict of model outputs
            targets: List of target dicts
            indices: List of (pred_idx, tgt_idx) indices for each batch
            num_masks: Number of matching masks
            
        Returns:
            Dict with classification loss
        """
        # Check if class prediction exists in the outputs
        if "pred_classes" not in outputs:
            return {"loss_classes": torch.as_tensor(0.0, device=self.device)}

        src_logits = outputs["pred_classes"]  # Shape: [batch_size, num_queries, num_classes]
        device = src_logits.device
        
        # Handle empty targets
        if len(targets) == 0 or all(len(t.get("classes", [])) == 0 for t in targets):
            loss = F.cross_entropy(
                src_logits.flatten(0, 1),
                torch.zeros(src_logits.shape[0] * src_logits.shape[1], dtype=torch.long, device=device),
                reduction="mean",
            )
            return {"loss_classes": loss * self.class_weight}
        
        focal_alpha = 0.25
        focal_gamma = 2.0
        
        # Initialize loss tensor
        loss = torch.tensor(0.0, device=device)
        
        # Process each image in the batch
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) == 0:  # Skip if no targets for this image
                continue
                
            # Get predictions for matched queries
            batch_src_logits = src_logits[batch_idx][src_idx]  # Shape: [num_matched, num_classes]
            
            # Check if 'classes' exists in the target
            if "classes" not in targets[batch_idx]:
                # If no classes, assume all are background (class 0)
                tgt_classes = torch.zeros(len(tgt_idx), dtype=torch.long, device=device)
            else:
                # Get target classes for matched ground truth
                tgt_classes = targets[batch_idx]["classes"][tgt_idx]
                
                # Ensure tgt_classes is a tensor with proper shape
                if not isinstance(tgt_classes, torch.Tensor):
                    tgt_classes = torch.tensor(tgt_classes, dtype=torch.long, device=device)
                elif len(tgt_classes.shape) == 0:
                    tgt_classes = tgt_classes.unsqueeze(0)

            # Apply focal loss
            probs = F.softmax(batch_src_logits, dim=-1)
            p_t = probs.gather(1, tgt_classes.unsqueeze(1)).squeeze(1)
            loss_batch = -focal_alpha * (1 - p_t) ** focal_gamma * torch.log(p_t + 1e-8)
            loss += loss_batch.sum()
        
        # Normalize loss by the number of matches
        if num_masks > 0:
            loss = loss / num_masks
        
        return {"loss_classes": loss * self.class_weight}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            "labels": self.loss_labels, 
            "masks": self.loss_masks,
            "classes": self.loss_classes
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=outputs["pred_logits"].device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
