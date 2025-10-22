# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from torch.cuda.amp import autocast


def batch_dice_loss(inputs, targets):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
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
    hw = inputs.shape[1]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum(
        "nc,mc->nm", focal_neg, (1 - targets)
    )

    return loss / hw


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, cost_class_prediction: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
            cost_class_prediction: This is the relative weight of the class prediction error in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_class_prediction = cost_class_prediction
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0 or cost_class_prediction != 0, "all costs cant be 0"

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        masks = [v["masks"] for v in targets]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            # Check if this is a negative sample (no instances to match)
            if targets[b].get("is_negative", False) or len(targets[b]["masks"]) == 0:
                # For negative samples, we don't need to do matching
                # Instead, we return empty indices to indicate no objects to match
                indices.append((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)))
                continue

            # out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            out_prob = outputs["pred_logits"][b] # [num_queries, num_classes]
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]

            tgt_ids = targets[b]["labels"]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask) # [num_total_targets, H_pred, W_pred]
            # print("tgt_mask shape:", tgt_mask.shape)

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            # cost_class = -out_prob[:, tgt_ids]
            cost_class = -out_prob[:, 0].unsqueeze(1)
            
            # Compute class prediction cost if available
            if "pred_classes" in outputs and self.cost_class_prediction > 0:
                out_class = outputs["pred_classes"][b]  # [num_queries, num_classes+1]
                tgt_class = targets[b]["classes"]  # [num_instances]
                
                # Convert logits to probabilities
                out_class_prob = F.softmax(out_class, dim=1)  # [num_queries, num_classes+1]
                
                # Apply focal loss approach similar to MaskDINO
                # Use focal loss parameters
                alpha = 0.25
                gamma = 2.0
                
                # Create cost matrix with correct shape
                num_targets = len(tgt_class)
                if num_targets == 0:
                    # Empty cost matrix when no targets
                    cost_class_pred = torch.zeros((num_queries, 1), device=out_class.device)
                    print("No target classes found, using dummy cost matrix")
                else:
                    # Initialize cost matrix
                    cost_class_pred = torch.zeros((num_queries, num_targets), device=out_class.device)
                    
                    for i, cls_id in enumerate(tgt_class):
                        # Get probability for the target class
                        p = out_class_prob[:, cls_id]
                        
                        # Focal loss formulation: -alpha * (1-p)^gamma * log(p)
                        # For matching, we want cost to be higher for wrong predictions, 
                        # so we use 1 - prob as the base cost
                        cost_class_pred[:, i] = alpha * ((1 - p) ** gamma) * (-(p + 1e-8).log())
            else:
                # If no class predictions available, use dummy cost
                cost_class_pred = torch.zeros_like(cost_class)

            # Downsample gt masks to save memory
            tgt_mask = F.interpolate(tgt_mask[:, None], size=out_mask.shape[-2:], mode="nearest")

            # Flatten spatial dimension
            out_mask = out_mask.flatten(1)  # [num_queries, H*W]
            # print("out_mask shape:", out_mask.shape)
            tgt_mask = tgt_mask[:, 0].flatten(1)  # [num_total_targets, H*W]
            # print("tgt_mask shape:", tgt_mask.shape)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask)
                cost_mask[cost_mask.isnan()] = 1e6

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss(out_mask, tgt_mask)
                cost_dice[cost_dice.isnan()] = 1e6

            # print("cost_mask shape:", cost_mask.shape)
            # print("cost_dice shape:", cost_dice.shape)
            # print("cost_class shape:", cost_class.shape)
            # Final cost matrix - ensure all cost matrices have the same shape
            # Check that the shapes match, and reshape or broadcast as needed
            num_tgt = cost_mask.shape[1]  # Number of targets
            
            # Ensure cost_class has the right shape
            if cost_class.shape[1] != num_tgt:
                if cost_class.shape[1] == 1:
                    # Broadcast to match number of targets
                    cost_class = cost_class.expand(-1, num_tgt)
                else:
                    # Fallback: create a zeros tensor of the right shape
                    cost_class = torch.zeros((num_queries, num_tgt), device=cost_mask.device)
            
            # Ensure cost_class_pred has the right shape
            if cost_class_pred.shape[1] != num_tgt:
                if cost_class_pred.shape[1] == 1:
                    # Broadcast to match number of targets
                    cost_class_pred = cost_class_pred.expand(-1, num_tgt)
                else:
                    # Fallback: resize the tensor to match targets
                    old_cost = cost_class_pred.clone()
                    cost_class_pred = torch.zeros((num_queries, num_tgt), device=cost_mask.device)
                    # Copy over values where possible
                    min_targets = min(old_cost.shape[1], num_tgt)
                    if min_targets > 0:
                        cost_class_pred[:, :min_targets] = old_cost[:, :min_targets]
            
            # Now combine the costs with their weights
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
                + self.cost_class_prediction * cost_class_pred
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
            "cost_class_prediction: {}".format(self.cost_class_prediction),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
