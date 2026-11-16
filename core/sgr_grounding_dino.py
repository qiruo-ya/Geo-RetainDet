"""Selective Generic Retention for strict-inductive OVAD training."""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from mmengine.utils import import_modules_from_strings
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.autograd import Function
from torchvision.ops import roi_align

from mmdet.models.dense_heads.atss_vlfusion_head import (
    convert_grounding_to_cls_scores)
from mmdet.models.dense_heads.deformable_detr_head import DeformableDETRHead
from mmdet.models.dense_heads.dino_head import DINOHead
from mmdet.models.dense_heads.grounding_dino_head import GroundingDINOHead
from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.models.losses import QualityFocalLoss
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
from mmdet.utils import ConfigType, InstanceList, OptInstanceList, reduce_mean


SGR_D1_PROMPTS = (
    'object',
    'foreground object',
    'physical entity',
    'man-made object',
)

NOVEL_CLASS_STRINGS = (
    'ground-track-field',
    'basketball-court',
    'swimming-pool',
    'helicopter',
)


class _ImplicitSoftBudget(Function):
    """Sigmoid soft Top-K with an implicitly differentiated threshold.

    The threshold is solved so that the memberships sum to ``budget``.  Its
    dependence on every priority is included in backward, which prevents a
    uniform score shift from changing the loss.  This implementation is kept
    local to the detector so the joint-training path does not depend on an
    offline experiment utility.
    """

    @staticmethod
    def forward(ctx, priority: Tensor, budget: int,
                temperature: float) -> Tensor:
        if priority.ndim != 1:
            raise ValueError('budget priority must be one-dimensional')
        count = int(priority.numel())
        k = min(int(budget), count)
        if k >= count:
            ctx.full = True
            return torch.ones_like(priority)
        ctx.full = False
        tau = float(temperature)
        with torch.no_grad():
            lower = priority.min() - 40.0 * tau
            upper = priority.max() + 40.0 * tau
            for _ in range(64):
                threshold = (lower + upper) * 0.5
                occupied = torch.sigmoid(
                    (priority - threshold) / tau).sum()
                lower, upper = (
                    torch.where(occupied > k, threshold, lower),
                    torch.where(occupied > k, upper, threshold))
            threshold = (lower + upper) * 0.5
            weights = torch.sigmoid((priority - threshold) / tau)
        ctx.temperature = tau
        ctx.save_for_backward(weights)
        return weights

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.full:
            return torch.zeros_like(grad_output), None, None
        (weights, ) = ctx.saved_tensors
        slope = weights * (1.0 - weights)
        denom = slope.sum().clamp_min(1e-12)
        centered = grad_output - (grad_output * slope).sum() / denom
        return slope * centered / ctx.temperature, None, None


def _soft_budget(priority: Tensor, budget: int,
                 temperature: float) -> Tensor:
    return _ImplicitSoftBudget.apply(
        priority, int(budget), float(temperature))


class ResidualClassificationAdapter(nn.Module):
    """Identity-initialized bottleneck adapter for classification features."""

    def __init__(self, embed_dims: int, bottleneck_dims: int) -> None:
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.bottleneck_dims = int(bottleneck_dims)
        if self.embed_dims <= 0 or self.bottleneck_dims <= 0:
            raise ValueError('classification adapter dimensions must be > 0')
        self.norm = nn.LayerNorm(self.embed_dims)
        self.down = nn.Linear(self.embed_dims, self.bottleneck_dims)
        self.up = nn.Linear(self.bottleneck_dims, self.embed_dims)
        self.reset_identity_parameters()

    def reset_identity_parameters(self) -> None:
        """Keep the pretrained scoring function exact at initialization."""
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = self.up(F.gelu(self.down(self.norm(hidden_states))))
        return hidden_states + residual


class GeoQualityHead(nn.Module):
    """Category-free two-scale spatial geometry reader used by GEO-V1."""

    def __init__(self, in_channels: int = 256, proj_channels: int = 32,
                 roi_size: int = 7) -> None:
        super().__init__()
        self.roi_size = int(roi_size)
        self.proj8 = nn.Conv2d(in_channels, proj_channels, 1)
        self.proj16 = nn.Conv2d(in_channels, proj_channels, 1)
        flat_dims = 2 * proj_channels * self.roi_size * self.roi_size
        self.norm = nn.LayerNorm(flat_dims)
        self.fc = nn.Linear(flat_dims, 256)
        self.delta = nn.Linear(256, 4)
        self.quality = nn.Linear(256, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.quality.weight)
        nn.init.zeros_(self.quality.bias)

    def forward(self, feat8: Tensor, feat16: Tensor, rois: Tensor,
                input_hw: tuple[int, int]) -> tuple[Tensor, Tensor]:
        input_h, input_w = input_hw
        scale8 = 0.5 * (feat8.shape[-2] / input_h +
                        feat8.shape[-1] / input_w)
        scale16 = 0.5 * (feat16.shape[-2] / input_h +
                         feat16.shape[-1] / input_w)
        x8 = roi_align(self.proj8(feat8), rois,
                       output_size=(self.roi_size, self.roi_size),
                       spatial_scale=float(scale8), sampling_ratio=2,
                       aligned=True)
        x16 = roi_align(self.proj16(feat16), rois,
                        output_size=(self.roi_size, self.roi_size),
                        spatial_scale=float(scale16), sampling_ratio=2,
                        aligned=True)
        hidden = F.gelu(self.fc(self.norm(
            torch.cat([x8, x16], dim=1).flatten(1))))
        return self.delta(hidden), self.quality(hidden).squeeze(-1).sigmoid()


class PrivateSpatialGeoAdapter(nn.Module):
    """Private spatial residual path used only by the GEO reader."""

    def __init__(self, channels: int = 256, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, channels, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: Tensor) -> Tensor:
        return features + self.net(features)


def _boxes_current_xyxy(bbox_pred: Tensor, img_meta: dict) -> Tensor:
    boxes = bbox_cxcywh_to_xyxy(bbox_pred)
    img_h, img_w = img_meta['img_shape']
    factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
    boxes = boxes * factor
    boxes[:, 0::2].clamp_(min=0, max=float(img_w))
    boxes[:, 1::2].clamp_(min=0, max=float(img_h))
    return boxes


def _bbox_overlaps_xyxy(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def _max_iou_to_gt(boxes: Tensor, gt_instances) -> Tensor:
    if len(gt_instances) == 0:
        return boxes.new_zeros((boxes.size(0), ))
    gt_bboxes = gt_instances.bboxes
    if hasattr(gt_bboxes, 'tensor'):
        gt_bboxes = gt_bboxes.tensor
    gt_bboxes = gt_bboxes.to(device=boxes.device, dtype=boxes.dtype)
    if gt_bboxes.numel() == 0:
        return boxes.new_zeros((boxes.size(0), ))
    overlaps = _bbox_overlaps_xyxy(boxes, gt_bboxes)
    return overlaps.max(dim=1).values if overlaps.numel() else \
        boxes.new_zeros((boxes.size(0), ))


def _normalize_xyxy(boxes: Tensor, img_meta: dict) -> Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    img_h, img_w = img_meta['img_shape']
    factor = boxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
    return (boxes / factor).clamp(0.0, 1.0)


def _boxes_original_to_current(boxes: Tensor, img_meta: dict) -> Tensor:
    """Transform cached original-image boxes into the current train image."""
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    out = boxes.clone()
    ori_shape = img_meta.get('ori_shape', img_meta.get('img_shape'))
    ori_h, ori_w = ori_shape[:2]
    if bool(img_meta.get('flip', False)):
        direction = img_meta.get('flip_direction', 'horizontal')
        if direction in ('horizontal', 'diagonal'):
            x1 = out[:, 0].clone()
            x2 = out[:, 2].clone()
            out[:, 0] = float(ori_w) - x2
            out[:, 2] = float(ori_w) - x1
        if direction in ('vertical', 'diagonal'):
            y1 = out[:, 1].clone()
            y2 = out[:, 3].clone()
            out[:, 1] = float(ori_h) - y2
            out[:, 3] = float(ori_h) - y1

    scale_factor = img_meta.get('scale_factor', None)
    if scale_factor is not None:
        sf = out.new_tensor(scale_factor).flatten()
        if sf.numel() == 2:
            sf = sf.repeat(2)
        elif sf.numel() != 4:
            raise AssertionError(f'Unexpected scale_factor: {scale_factor}')
        out = out * sf[None, :]

    img_h, img_w = img_meta['img_shape']
    out[:, 0::2].clamp_(min=0, max=float(img_w))
    out[:, 1::2].clamp_(min=0, max=float(img_h))
    return out


@MODELS.register_module()
class SGRGroundingDINOHead(GroundingDINOHead):
    """GroundingDINO head with UBP support on matching queries only."""

    def __init__(self,
                 sgr_alpha_bg: float = 1.0,
                 sgr_audit_ubp_delta: bool = False,
                 sgr_ubp_scope: str = 'all_decoder',
                 cls_adapter: Optional[ConfigType] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.sgr_alpha_bg = float(sgr_alpha_bg)
        self.sgr_audit_ubp_delta = bool(sgr_audit_ubp_delta)
        self.sgr_ubp_scope = str(sgr_ubp_scope)
        if self.sgr_ubp_scope not in ('all_decoder', 'last_decoder'):
            raise ValueError(
                'sgr_ubp_scope must be all_decoder or last_decoder, '
                f'got {self.sgr_ubp_scope!r}')
        self._sgr_protected_mask: Optional[Tensor] = None
        self._sgr_protected_weights: Optional[Tensor] = None
        self._sgr_apply_ubp = False
        self._sgr_current_decoder_layer: Optional[int] = None
        self._sgr_ubp_audit_rows: list[dict[str, float]] = []
        adapter_cfg = dict(cls_adapter or {})
        self.cls_adapter_enabled = bool(adapter_cfg.get('enabled', False))
        self.cls_adapter_bottleneck_dims = int(
            adapter_cfg.get('bottleneck_dims', 64))
        if self.cls_adapter_enabled:
            self.cls_adapter = ResidualClassificationAdapter(
                embed_dims=int(self.embed_dims),
                bottleneck_dims=self.cls_adapter_bottleneck_dims)
        else:
            self.cls_adapter = None

    def init_weights(self) -> None:
        super().init_weights()
        if self.cls_adapter is not None:
            self.cls_adapter.reset_identity_parameters()

    def forward_cls(self, branch_idx: int, visual_feat: Tensor,
                    text_feat: Tensor, text_token_mask: Tensor) -> Tensor:
        if self.cls_adapter is not None:
            visual_feat = self.cls_adapter(visual_feat)
        return super().forward_cls(branch_idx, visual_feat, text_feat,
                                   text_token_mask)

    @staticmethod
    def _get_num_dn_queries(dn_meta: Dict[str, int] | None) -> int:
        if dn_meta is None:
            return 0
        return int(dn_meta.get('num_denoising_queries', 0))

    @torch.no_grad()
    def get_base_pos_mask(self, cls_scores: Tensor, bbox_preds: Tensor,
                          batch_gt_instances: InstanceList,
                          batch_img_metas: List[dict]) -> Tensor:
        masks = []
        for cls_score, bbox_pred, gt_instances, img_meta in zip(
                cls_scores, bbox_preds, batch_gt_instances, batch_img_metas):
            if len(gt_instances) == 0:
                masks.append(
                    torch.zeros(
                        bbox_pred.size(0),
                        dtype=torch.bool,
                        device=bbox_pred.device))
                continue
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0)
            pred_instances = InstanceData(
                scores=cls_score,
                bboxes=bbox_cxcywh_to_xyxy(bbox_pred) * factor)
            assign_result = self.assigner.assign(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta)
            masks.append(assign_result.gt_inds > 0)
        return torch.stack(masks, dim=0)

    def loss(self,
             hidden_states: Tensor,
             references: List[Tensor],
             memory_text: Tensor,
             text_token_mask: Tensor,
             enc_outputs_class: Tensor,
             enc_outputs_coord: Tensor,
             batch_data_samples: SampleList,
             dn_meta: Dict[str, int],
             sgr_protected_mask: Optional[Tensor] = None,
             sgr_protected_weights: Optional[Tensor] = None) -> dict:
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references, memory_text, text_token_mask)
        self.text_masks = text_token_mask
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        return self.loss_by_feat(
            *loss_inputs,
            sgr_protected_mask=sgr_protected_mask,
            sgr_protected_weights=sgr_protected_weights)

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None,
        sgr_protected_mask: Optional[Tensor] = None,
        sgr_protected_weights: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_bbox_preds, dn_meta)

        old_mask = self._sgr_protected_mask
        old_weights = self._sgr_protected_weights
        old_apply = self._sgr_apply_ubp
        old_layer = self._sgr_current_decoder_layer
        self._sgr_ubp_audit_rows = []
        self._sgr_protected_mask = sgr_protected_mask
        self._sgr_protected_weights = sgr_protected_weights
        self._sgr_apply_ubp = sgr_protected_mask is not None
        try:
            if self.sgr_ubp_scope == 'all_decoder':
                # Preserve the historical parent implementation exactly for
                # old configs and checkpoints.
                loss_dict = super(DeformableDETRHead, self).loss_by_feat(
                    all_layers_matching_cls_scores,
                    all_layers_matching_bbox_preds,
                    batch_gt_instances,
                    batch_img_metas,
                    batch_gt_instances_ignore)
                if self.sgr_audit_ubp_delta:
                    for layer_idx, row in enumerate(
                            self._sgr_ubp_audit_rows):
                        row['decoder_layer'] = float(layer_idx)
            else:
                assert batch_gt_instances_ignore is None, \
                    f'{self.__class__.__name__} only supports ' \
                    'batch_gt_instances_ignore=None.'
                losses_cls = []
                losses_bbox = []
                losses_iou = []
                last_layer = len(all_layers_matching_cls_scores) - 1
                for layer_idx, (cls_scores, bbox_preds) in enumerate(zip(
                        all_layers_matching_cls_scores,
                        all_layers_matching_bbox_preds)):
                    self._sgr_current_decoder_layer = layer_idx
                    self._sgr_apply_ubp = (
                        sgr_protected_mask is not None
                        and layer_idx == last_layer)
                    loss_cls, loss_bbox, loss_iou = self.loss_by_feat_single(
                        cls_scores,
                        bbox_preds,
                        batch_gt_instances=batch_gt_instances,
                        batch_img_metas=batch_img_metas)
                    losses_cls.append(loss_cls)
                    losses_bbox.append(loss_bbox)
                    losses_iou.append(loss_iou)
                loss_dict = dict(
                    loss_cls=losses_cls[-1],
                    loss_bbox=losses_bbox[-1],
                    loss_iou=losses_iou[-1])
                for layer_idx, (loss_cls, loss_bbox, loss_iou) in enumerate(
                        zip(losses_cls[:-1], losses_bbox[:-1],
                            losses_iou[:-1])):
                    loss_dict[f'd{layer_idx}.loss_cls'] = loss_cls
                    loss_dict[f'd{layer_idx}.loss_bbox'] = loss_bbox
                    loss_dict[f'd{layer_idx}.loss_iou'] = loss_iou
        finally:
            self._sgr_protected_mask = old_mask
            self._sgr_protected_weights = old_weights
            self._sgr_apply_ubp = old_apply
            self._sgr_current_decoder_layer = old_layer

        if enc_cls_scores is not None:
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores,
                    enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        if all_layers_denoising_cls_scores is not None:
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1],
                                  dn_losses_iou[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
        return loss_dict

    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> tuple:
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        with torch.no_grad():
            cls_reg_targets = self.get_targets(cls_scores_list,
                                               bbox_preds_list,
                                               batch_gt_instances,
                                               batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        audit_ubp = bool(self.sgr_audit_ubp_delta and self._sgr_apply_ubp)
        unmasked_label_weights = label_weights.clone() if audit_ubp else None
        protected_negative_count = 0
        removed_label_weight = 0.0
        if (self._sgr_apply_ubp and self._sgr_protected_mask is not None
                and (self.sgr_alpha_bg != 1.0
                     or self._sgr_protected_weights is not None)):
            protected = self._sgr_protected_mask.to(
                device=label_weights.device)
            if tuple(protected.shape) != tuple(label_weights.shape):
                raise AssertionError(
                    'SGR protected mask shape must match matching queries: '
                    f'{tuple(protected.shape)} vs {tuple(label_weights.shape)}')
            positive = labels.sum(dim=-1) > 0
            protected = protected & ~positive
            protected_negative_count = int(protected.sum().detach().item())
            label_weights = label_weights.clone()
            if self._sgr_protected_weights is None:
                alpha_bg = label_weights.new_full(
                    label_weights.shape, float(self.sgr_alpha_bg))
            else:
                protect_weights = self._sgr_protected_weights.to(
                    device=label_weights.device, dtype=label_weights.dtype)
                if tuple(protect_weights.shape) != tuple(
                        label_weights.shape):
                    raise AssertionError(
                        'SGR protected weights shape must match matching '
                        f'queries: {tuple(protect_weights.shape)} vs '
                        f'{tuple(label_weights.shape)}')
                if bool(((protect_weights < 0) |
                         (protect_weights > 1)).any().detach()):
                    raise AssertionError(
                        'SGR protected weights must lie in [0, 1].')
                alpha_bg = 1.0 - protect_weights
            label_weights[protected] = (
                label_weights[protected] * alpha_bg[protected])
            if unmasked_label_weights is not None:
                removed_label_weight = float(
                    (unmasked_label_weights -
                     label_weights).detach().sum().item())

        assert (self.text_masks.dim() == 2)
        text_masks = self.text_masks.new_zeros(
            (self.text_masks.size(0), self.max_text_len))
        text_masks[:, :self.text_masks.size(1)] = self.text_masks
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, cls_scores.size(1), 1)
        cls_scores = torch.masked_select(cls_scores, text_mask).contiguous()

        labels = torch.masked_select(labels, text_mask)
        if unmasked_label_weights is not None:
            unmasked_label_weights = unmasked_label_weights[..., None].repeat(
                1, 1, text_mask.size(-1))
            unmasked_label_weights = torch.masked_select(
                unmasked_label_weights, text_mask)
        label_weights = label_weights[..., None].repeat(
            1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)

        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            raise NotImplementedError(
                'QualityFocalLoss for GroundingDINOHead is not supported yet.')
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        if audit_ubp:
            assert unmasked_label_weights is not None
            loss_cls_unmasked = self.loss_cls(
                cls_scores,
                labels,
                unmasked_label_weights,
                avg_factor=cls_avg_factor)
            ubp_delta = loss_cls - loss_cls_unmasked
            ubp_grad = torch.autograd.grad(
                ubp_delta,
                cls_scores,
                retain_graph=True,
                allow_unused=True)[0]
            ubp_grad_norm = 0.0 if ubp_grad is None else float(
                ubp_grad.detach().float().norm().cpu())
            self._sgr_ubp_audit_rows.append(
                dict(
                    decoder_layer=(-1.0 if self._sgr_current_decoder_layer
                                   is None else float(
                                       self._sgr_current_decoder_layer)),
                    ubp_applied=float(
                        self._sgr_apply_ubp and
                        (self.sgr_alpha_bg != 1.0 or
                         self._sgr_protected_weights is not None)),
                    protected_negative_count=float(protected_negative_count),
                    removed_label_weight=float(removed_label_weight),
                    loss_cls_masked=float(loss_cls.detach().float().cpu()),
                    loss_cls_unmasked=float(
                        loss_cls_unmasked.detach().float().cpu()),
                    ubp_removed_loss=float(ubp_delta.detach().float().cpu()),
                    ubp_removed_grad_norm_logits=ubp_grad_norm))

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou


@MODELS.register_module()
class SelectiveGenericRetentionGroundingDINO(GroundingDINO):
    """Grounding DINO student with SGR teacher protection and aux losses."""

    def __init__(self,
                 *args,
                 sgr_cfg: Optional[ConfigType] = None,
                 **kwargs) -> None:
        self.sgr_cfg = dict(sgr_cfg or {})
        # LAE-DINO's Dynamic Vocabulary Construction (DVC) is needed when a
        # constituent dataset exposes hundreds of category names.  Zero keeps
        # every historical DOTA/DIOR run bit-for-bit on the old path.
        self.sgr_dvc_max_classes = int(
            self.sgr_cfg.get('dvc_max_classes', 0))
        if self.sgr_dvc_max_classes < 0:
            raise ValueError('sgr_cfg.dvc_max_classes must be non-negative')
        self.sgr_enabled = bool(self.sgr_cfg.get('enabled', True))
        self.sgr_alpha_bg = float(self.sgr_cfg.get('alpha_bg', 1.0))
        self.sgr_ubp_scope = str(
            self.sgr_cfg.get('ubp_scope', 'all_decoder'))
        if self.sgr_ubp_scope not in ('all_decoder', 'last_decoder'):
            raise ValueError(
                'sgr_cfg.ubp_scope must be all_decoder or last_decoder, '
                f'got {self.sgr_ubp_scope!r}')
        self.sgr_lambda_gfr = float(self.sgr_cfg.get('lambda_gfr', 0.0))
        self.sgr_lambda_grr = float(self.sgr_cfg.get('lambda_grr', 0.0))
        self.sgr_lambda_rank = float(self.sgr_cfg.get('lambda_rank', 0.0))
        self.sgr_lambda_scr = float(self.sgr_cfg.get('lambda_scr', 0.0))
        self.sgr_lambda_crsb = float(self.sgr_cfg.get('lambda_crsb', 0.0))
        self.sgr_lambda_rvrd = float(self.sgr_cfg.get('lambda_rvrd', 0.0))
        # Joint budget-coverage control.  A zero coefficient leaves the
        # historical model and inference graph unchanged.  The nonzero path
        # adds a training-only loss on the existing final decoder scores; it
        # deliberately introduces no new parameter or inference head.
        self.sgr_lambda_budget = float(
            self.sgr_cfg.get('lambda_budget', 0.0))
        geo_cfg = dict(self.sgr_cfg.get('geo_cfg', {}))
        self.geo_enabled = bool(geo_cfg.get('enabled', False))
        self.geo_detach_visual = bool(geo_cfg.get('detach_visual', False))
        self.geo_grad_levels = tuple(int(x) for x in
                                     geo_cfg.get('grad_levels', (0, 1)))
        if (not self.geo_grad_levels or
                any(x not in (0, 1) for x in self.geo_grad_levels)):
            raise ValueError('geo_cfg.grad_levels must be a non-empty subset of (0, 1)')
        self.geo_lambda = float(geo_cfg.get('lambda_geo', 0.1))
        # Defaults preserve the original GOL objective.  The quality-only
        # ablation keeps the same head and sampling, disabling auxiliaries only.
        self.geo_quality_weight = float(geo_cfg.get('quality_weight', 1.0))
        self.geo_offset_weight = float(geo_cfg.get('offset_weight', 1.0))
        self.geo_giou_weight = float(geo_cfg.get('giou_weight', 1.0))
        if any(not math.isfinite(x) or x < 0 for x in (
                self.geo_quality_weight, self.geo_offset_weight,
                self.geo_giou_weight)):
            raise ValueError('GOL loss component weights must be finite and nonnegative')
        self.geo_warmup_iters = int(geo_cfg.get('warmup_iters', 1669))
        self.geo_max_gt = int(geo_cfg.get('max_gt_per_image', 16))
        self.geo_max_real = int(geo_cfg.get('max_real_per_image', 64))
        self.geo_roi_scale = float(geo_cfg.get('roi_scale', 1.25))
        self.geo_seed = int(geo_cfg.get('seed', 3407))
        self.geo_private_adapter = bool(
            geo_cfg.get('private_spatial_adapter', False))
        self.geo_adapter_hidden = int(geo_cfg.get('adapter_hidden', 64))
        self.geo_freeze_detector = bool(
            geo_cfg.get('freeze_detector', False))
        self.geo_train_visual_only = bool(
            geo_cfg.get('train_visual_only', False))
        if self.geo_freeze_detector and self.geo_train_visual_only:
            raise ValueError('freeze_detector and train_visual_only conflict')
        box_cfg = dict(self.sgr_cfg.get('novel_box_cfg', {}))
        self.novel_box_enabled = bool(box_cfg.get('enabled', False))
        self.novel_box_bbox_weight = float(box_cfg.get('bbox_weight', 5.0))
        self.novel_box_giou_weight = float(box_cfg.get('giou_weight', 2.0))
        keep_cfg = dict(self.sgr_cfg.get('keep_cfg', {}))
        self.keep_enabled = bool(keep_cfg.get('enabled', False))
        self.keep_lambda = float(keep_cfg.get('lambda_keep', 1.0))
        self.keep_topk = int(keep_cfg.get('topk', 900))
        self.keep_teacher_config = keep_cfg.get('teacher_config', None)
        self.keep_teacher_checkpoint = keep_cfg.get('teacher_checkpoint', None)
        self._keep_student_dense_cls: Optional[Tensor] = None
        self._keep_student_memory_mask: Optional[Tensor] = None
        self._last_keep_stats: dict[str, Any] = {}
        self._last_novel_box_stats: dict[str, Any] = {}
        self.sgr_budget = int(self.sgr_cfg.get('budget', 50))
        self.sgr_budget_temperature = float(
            self.sgr_cfg.get('budget_temperature', 0.1))
        self.sgr_budget_iou_thr = float(
            self.sgr_cfg.get('budget_iou_thr', 0.5))
        self.sgr_budget_warmup_iters = int(
            self.sgr_cfg.get('budget_warmup_iters', 0))
        if self.sgr_budget <= 0:
            raise ValueError('sgr_cfg.budget must be positive')
        if self.sgr_budget_temperature <= 0:
            raise ValueError('sgr_cfg.budget_temperature must be positive')
        if not 0.0 <= self.sgr_budget_iou_thr <= 1.0:
            raise ValueError('sgr_cfg.budget_iou_thr must lie in [0, 1]')
        self.sgr_score_thr = float(self.sgr_cfg.get('score_thr', 0.106708481908))
        self.sgr_stability_thr = float(self.sgr_cfg.get('stability_thr', 0.7))
        self.sgr_base_iou_thr = float(self.sgr_cfg.get('base_iou_thr', 0.5))
        self.sgr_max_protected = int(self.sgr_cfg.get('max_protected', 16))
        self.sgr_match_iou_thr = float(self.sgr_cfg.get('match_iou_thr', 0.3))
        self.sgr_grr_negatives = int(self.sgr_cfg.get('grr_negatives', 4))
        self.sgr_grr_margin = float(self.sgr_cfg.get('grr_margin', 0.1))
        self.sgr_grr_temperature = float(
            self.sgr_cfg.get('grr_temperature', 0.05))
        self.sgr_rank_negatives = int(
            self.sgr_cfg.get('rank_negatives', 4))
        self.sgr_rank_margin = float(self.sgr_cfg.get('rank_margin', 0.1))
        self.sgr_soft_ubp_enabled = bool(
            self.sgr_cfg.get('soft_ubp_enabled', False))
        self.sgr_generic_temperature = float(
            self.sgr_cfg.get('generic_temperature', 1.0))
        self.sgr_scr_temperature = float(
            self.sgr_cfg.get('scr_temperature', 1.0))
        self.sgr_scr_eps = float(self.sgr_cfg.get('scr_eps', 1e-6))
        self.sgr_scr_weight_min = float(
            self.sgr_cfg.get('scr_weight_min', 0.25))
        self.sgr_scr_weight_max = float(
            self.sgr_cfg.get('scr_weight_max', 4.0))
        self.sgr_scr_weight_mode = str(
            self.sgr_cfg.get('scr_weight_mode', 'js_relative'))
        if self.sgr_scr_weight_mode not in ('js_relative', 'uniform'):
            raise ValueError(
                'sgr_cfg.scr_weight_mode must be js_relative or uniform, '
                f'got {self.sgr_scr_weight_mode!r}')
        self.sgr_scr_target_mode = str(
            self.sgr_cfg.get('scr_target_mode', 'full'))
        if self.sgr_scr_target_mode not in ('full', 'base_collapsed'):
            raise ValueError(
                'sgr_cfg.scr_target_mode must be full or base_collapsed, '
                f'got {self.sgr_scr_target_mode!r}')
        self.sgr_scr_log_decomposition = bool(
            self.sgr_cfg.get('scr_log_decomposition', False))
        self.sgr_scr_audit_tensors = bool(
            self.sgr_cfg.get('scr_audit_tensors', False))
        # Audit-only graph references for R009.  The default is false and the
        # references are never serialized, so ordinary training/inference is
        # unchanged.  Keeping the original transformer output (rather than a
        # view created after the classification head) is essential: both the
        # detector score and SCR must be descendants of the same graph node.
        self.sgr_scr_direction_audit_tensors = bool(
            self.sgr_cfg.get('scr_direction_audit_tensors', False))
        self.sgr_log_layer_audit = bool(
            self.sgr_cfg.get('log_layer_audit', False))
        self.sgr_crsb_cache_file = self.sgr_cfg.get('crsb_cache_file', None)
        self.sgr_crsb_beta_max = float(
            self.sgr_cfg.get('crsb_beta_max', 0.0))
        self.sgr_crsb_tau_g = float(
            self.sgr_cfg.get('crsb_tau_g', self.sgr_scr_temperature))
        self.sgr_crsb_tau_r = float(self.sgr_cfg.get('crsb_tau_r', 1.0))
        self.sgr_crsb_delta_peak = float(
            self.sgr_cfg.get('crsb_delta_peak', 0.05))
        self.sgr_crsb_cache_match_iou = float(
            self.sgr_cfg.get('crsb_cache_match_iou', 0.5))
        self.sgr_crsb_guard_bypass_when_beta_zero = bool(
            self.sgr_cfg.get('crsb_guard_bypass_when_beta_zero', True))
        self.sgr_peak_base_guard_enabled = bool(
            self.sgr_cfg.get('peak_base_guard_enabled', True))
        self.sgr_warmup_iters = int(self.sgr_cfg.get('warmup_iters', 1000))
        self.sgr_rvrd_cache_file = self.sgr_cfg.get('rvrd_cache_file', None)
        self.sgr_rvrd_teacher = str(self.sgr_cfg.get('rvrd_teacher', ''))
        self.sgr_rvrd_cache_match_iou = float(
            self.sgr_cfg.get('rvrd_cache_match_iou',
                             self.sgr_crsb_cache_match_iou))
        self.sgr_rvrd_temperature = float(
            self.sgr_cfg.get('rvrd_temperature', 0.10))
        self.sgr_rvrd_warmup_iters = int(
            self.sgr_cfg.get('rvrd_warmup_iters', self.sgr_warmup_iters))
        self.sgr_rvrd_min_queries = int(
            self.sgr_cfg.get('rvrd_min_queries', 8))
        self.sgr_rvrd_duplicate_iou = float(
            self.sgr_cfg.get('rvrd_duplicate_iou', 0.7))
        self.sgr_rvrd_expected_images = int(
            self.sgr_cfg.get('rvrd_expected_images', 0))
        self.sgr_rvrd_max_zero_streak = int(
            self.sgr_cfg.get('rvrd_max_zero_streak', 50))
        self.sgr_rvrd_grad_audit_interval = int(
            self.sgr_cfg.get('rvrd_grad_audit_interval', 0))
        self.sgr_prompt_set = str(self.sgr_cfg.get('teacher_prompt_set', 'D1'))
        self.sgr_prompts = tuple(
            self.sgr_cfg.get('generic_prompts', SGR_D1_PROMPTS))
        self.sgr_semantic_anchor_prompts = tuple(
            self.sgr_cfg.get('semantic_anchor_prompts', ()))
        self.sgr_semantic_base_count = int(
            self.sgr_cfg.get('semantic_base_count', 0))
        # Keep the historical split as the default, but make the leakage
        # guard protocol-configurable for other registered DOTA/DIOR splits.
        self.sgr_novel_class_strings = tuple(
            self.sgr_cfg.get('novel_class_strings', NOVEL_CLASS_STRINGS))
        self.sgr_teacher_config = self.sgr_cfg.get('teacher_config', None)
        self.sgr_teacher_checkpoint = self.sgr_cfg.get('teacher_checkpoint',
                                                       None)
        self.sgr_teacher_no_grad = bool(
            self.sgr_cfg.get('teacher_no_grad', True))
        self.sgr_build_teacher_zero_effect = bool(
            self.sgr_cfg.get('build_teacher_zero_effect', True))
        self.sgr_debug_stats = bool(self.sgr_cfg.get('debug_stats', False))
        self.sgr_log_loss_keys = bool(self.sgr_cfg.get('log_loss_keys', False))
        self.sgr_log_scr_keys = bool(self.sgr_cfg.get('log_scr_keys', False))
        self.sgr_log_crsb_keys = bool(
            self.sgr_cfg.get('log_crsb_keys', False))
        self.sgr_log_rvrd_keys = bool(
            self.sgr_cfg.get('log_rvrd_keys', False))
        self._sgr_teacher = None
        self._sgr_generic_text_cache: Optional[dict[str, Any]] = None
        self._sgr_anchor_text_cache: Optional[dict[str, Any]] = None
        self._sgr_teacher_anchor_text_cache: Optional[dict[str, Any]] = None
        self._sgr_crsb_cache: Optional[dict[str, Any]] = None
        self._sgr_rvrd_cache: Optional[dict[str, Any]] = None
        self._sgr_rvrd_zero_streak = 0
        self._sgr_iter = 0
        self._last_sgr_stats: dict[str, Any] = {}
        self._last_sgr_protected_mask: Optional[Tensor] = None
        self._last_scr_audit_tensors: dict[str, Tensor] = {}
        self._last_budget_stats: dict[str, Any] = {}
        if self.sgr_prompt_set != 'D1':
            raise ValueError('The registered SGR training path currently '
                             'expects D1 generic prompts.')
        prompt_text = ' '.join(self.sgr_prompts +
                               self.sgr_semantic_anchor_prompts)
        if any(name in prompt_text for name in self.sgr_novel_class_strings):
            raise AssertionError('SGR generic prompts contain novel strings.')
        if (self.sgr_lambda_scr != 0.0 or self.sgr_lambda_crsb != 0.0
                or self.sgr_log_scr_keys or self.sgr_log_crsb_keys):
            if not self.sgr_semantic_anchor_prompts:
                raise ValueError('DSCR/CRSB requires semantic_anchor_prompts.')
            if not (0 < self.sgr_semantic_base_count <=
                    len(self.sgr_semantic_anchor_prompts)):
                raise ValueError('semantic_base_count must be within the '
                                 'semantic anchor prompt list.')
        if (self.sgr_lambda_crsb != 0.0 and self.sgr_crsb_beta_max != 0.0
                and self.sgr_crsb_cache_file is None):
            raise ValueError('CRSB beta>0 requires sgr_cfg.crsb_cache_file.')
        if self.sgr_lambda_rvrd != 0.0 and self.sgr_rvrd_cache_file is None:
            raise ValueError('RVRD requires sgr_cfg.rvrd_cache_file.')
        super().__init__(*args, **kwargs)
        if self.geo_enabled:
            # Do not perturb the detector/data-loader RNG trajectory.  GEO
            # initialization has its own deterministic CPU RNG scope.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.geo_seed + 7919)
                self.geo_head = GeoQualityHead()
                if self.geo_private_adapter:
                    self.geo_adapters = nn.ModuleList([
                        PrivateSpatialGeoAdapter(
                            hidden=self.geo_adapter_hidden),
                        PrivateSpatialGeoAdapter(
                            hidden=self.geo_adapter_hidden)])
                else:
                    self.geo_adapters = None
        else:
            self.geo_head = None
            self.geo_adapters = None
        if isinstance(self.bbox_head, SGRGroundingDINOHead):
            self.bbox_head.sgr_alpha_bg = self.sgr_alpha_bg
            self.bbox_head.sgr_ubp_scope = self.sgr_ubp_scope
            self.bbox_head.sgr_audit_ubp_delta = bool(
                self.sgr_cfg.get('audit_ubp_delta', False))
        elif self.sgr_alpha_bg != 1.0 or self.sgr_soft_ubp_enabled:
            raise TypeError('UBP requires SGRGroundingDINOHead.')
        self._configure_geo_trainable_parameters()

    def _configure_geo_trainable_parameters(self) -> None:
        """Apply N1/N2 trainable-parameter policies before optimizer build."""
        if not (self.geo_freeze_detector or self.geo_train_visual_only):
            return
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if self.geo_train_visual_only:
            for module in (self.backbone, self.neck):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        for module in (self.geo_head, self.geo_adapters):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.geo_freeze_detector:
            for name, module in self.named_children():
                if name not in ('geo_head', 'geo_adapters'):
                    module.train(False)
            if self.geo_head is not None:
                self.geo_head.train(mode)
            if self.geo_adapters is not None:
                self.geo_adapters.train(mode)
        return self

    def forward_transformer(self, img_feats, text_dict,
                            batch_data_samples=None) -> Dict:
        """Expose dense encoder responses only for the N2 keep objective."""
        if not self.keep_enabled:
            return super().forward_transformer(
                img_feats, text_dict, batch_data_samples)
        encoder_inputs, decoder_inputs = self.pre_transformer(
            img_feats, batch_data_samples)
        encoded = self.forward_encoder(**encoder_inputs, text_dict=text_dict)
        output_memory, _ = self.gen_encoder_output_proposals(
            encoded['memory'], encoded['memory_mask'],
            encoded['spatial_shapes'])
        dense = self.bbox_head.forward_cls(
            self.decoder.num_layers, output_memory,
            encoded['memory_text'], encoded['text_token_mask'])
        # Call the exact parent implementation so query selection remains
        # byte-for-byte the BaseFT path; the extra dense tensor is diagnostic.
        tmp_dec, head_inputs = GroundingDINO.pre_decoder(
            self, **encoded, batch_data_samples=batch_data_samples)
        decoder_inputs.update(tmp_dec)
        head_inputs.update(self.forward_decoder(**decoder_inputs))
        head_inputs['keep_student_dense_cls'] = dense
        head_inputs['keep_student_memory_mask'] = encoded['memory_mask']
        # Keep an explicit per-forward reference as well.  The loss path uses
        # only this same-iteration tensor; storing it here avoids relying on
        # auxiliary keys surviving wrapper/dict handling before the keep loss
        # is evaluated, while preserving its autograd graph.
        self._keep_student_dense_cls = dense
        self._keep_student_memory_mask = encoded['memory_mask']
        return head_inputs

    def pre_decoder(self, memory: Tensor, memory_mask: Tensor,
                    spatial_shapes: Tensor, memory_text: Tensor,
                    text_token_mask: Tensor,
                    batch_data_samples=None):
        """Cache dense student responses for task-adapted keep loss."""
        if self.keep_enabled:
            output_memory, _ = self.gen_encoder_output_proposals(
                memory, memory_mask, spatial_shapes)
            self._keep_student_dense_cls = self.bbox_head.forward_cls(
                self.decoder.num_layers, output_memory, memory_text,
                text_token_mask)
            self._keep_student_memory_mask = memory_mask
        else:
            self._keep_student_dense_cls = None
            self._keep_student_memory_mask = None
        decoder_inputs, head_inputs = super().pre_decoder(
            memory=memory, memory_mask=memory_mask,
            spatial_shapes=spatial_shapes, memory_text=memory_text,
            text_token_mask=text_token_mask,
            batch_data_samples=batch_data_samples)
        if self.keep_enabled:
            head_inputs['keep_student_dense_cls'] = \
                self._keep_student_dense_cls
            head_inputs['keep_student_memory_mask'] = \
                self._keep_student_memory_mask
        return decoder_inputs, head_inputs

    def _geo_input_features(self, visual_features) -> tuple[Tensor, Tensor]:
        feat8, feat16 = visual_features[0], visual_features[1]
        if self.geo_private_adapter:
            if self.geo_adapters is None:
                raise AssertionError('private GEO adapter was not built')
            if self.geo_freeze_detector or self.geo_detach_visual:
                feat8, feat16 = feat8.detach(), feat16.detach()
            else:
                if 0 not in self.geo_grad_levels:
                    feat8 = feat8.detach()
                if 1 not in self.geo_grad_levels:
                    feat16 = feat16.detach()
            return self.geo_adapters[0](feat8), self.geo_adapters[1](feat16)
        if self.geo_detach_visual:
            return feat8.detach(), feat16.detach()
        if 0 not in self.geo_grad_levels:
            feat8 = feat8.detach()
        if 1 not in self.geo_grad_levels:
            feat16 = feat16.detach()
        return feat8, feat16

    @torch.no_grad()
    def _forward_keep_teacher_dense(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> tuple[Tensor, Tensor, list]:
        if self.keep_teacher_config is None or self.keep_teacher_checkpoint is None:
            raise ValueError('keep loss requires BaseFT teacher config/checkpoint')
        # Reuse the lazy frozen teacher container already audited by SCR.
        old_cfg, old_ckpt = self.sgr_teacher_config, self.sgr_teacher_checkpoint
        self.sgr_teacher_config = self.keep_teacher_config
        self.sgr_teacher_checkpoint = self.keep_teacher_checkpoint
        try:
            teacher = self._get_sgr_teacher(batch_inputs.device)
        finally:
            self.sgr_teacher_config, self.sgr_teacher_checkpoint = old_cfg, old_ckpt
        normalized, token_maps = [], []
        for sample in batch_data_samples:
            token_map, prompt, _, _ = teacher.get_tokens_positive_and_prompts(
                sample.text, bool(sample.custom_entities)
                if 'custom_entities' in sample else False,
                sample.caption_prompt if 'caption_prompt' in sample else None,
                sample.get('tokens_positive', None))
            if isinstance(prompt, list):
                raise NotImplementedError('chunked keep prompts unsupported')
            normalized.append(prompt); token_maps.append(token_map)
        text_dict = teacher.language_model(normalized)
        if teacher.text_feat_map is not None:
            text_dict['embedded'] = teacher.text_feat_map(text_dict['embedded'])
        visual = teacher.extract_feat(batch_inputs)
        encoder_inputs, _ = teacher.pre_transformer(
            visual, batch_data_samples)
        encoded = teacher.forward_encoder(**encoder_inputs, text_dict=text_dict)
        output_memory, _ = teacher.gen_encoder_output_proposals(
            encoded['memory'], encoded['memory_mask'], encoded['spatial_shapes'])
        dense = teacher.bbox_head.forward_cls(
            teacher.decoder.num_layers, output_memory,
            encoded['memory_text'], encoded['text_token_mask'])
        return dense, encoded['memory_mask'], token_maps

    def _loss_task_adapted_keep(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> tuple[Tensor, dict[str, Any]]:
        student_dense = self._keep_student_dense_cls
        student_mask = self._keep_student_memory_mask
        if student_dense is None:
            raise AssertionError('student dense encoder responses missing')
        teacher_dense, teacher_mask, teacher_maps = \
            self._forward_keep_teacher_dense(batch_inputs, batch_data_samples)
        # Grounding DINO configurations with ``pad_mask=False`` legitimately
        # return no encoder padding mask.  Treat that as an all-valid spatial
        # sequence instead of confusing it with a missing dense response.
        if student_mask is None:
            student_mask = torch.zeros(
                student_dense.shape[:2], dtype=torch.bool,
                device=student_dense.device)
        if teacher_mask is None:
            teacher_mask = torch.zeros(
                teacher_dense.shape[:2], dtype=torch.bool,
                device=teacher_dense.device)
        student_maps = self._budget_class_maps(batch_data_samples)
        student_probs = convert_grounding_to_cls_scores(
            logits=student_dense.sigmoid(), positive_maps=student_maps)
        teacher_probs = convert_grounding_to_cls_scores(
            logits=teacher_dense.sigmoid(), positive_maps=teacher_maps).detach()
        if student_probs.shape != teacher_probs.shape:
            raise AssertionError('teacher/student dense response shape mismatch')
        if student_mask.shape != teacher_mask.shape:
            raise AssertionError('teacher/student dense mask shape mismatch')
        per_image = []
        chosen = 0
        for image_idx in range(student_probs.size(0)):
            objectness = teacher_probs[image_idx].max(dim=-1).values
            objectness = objectness.masked_fill(teacher_mask[image_idx], -1.0)
            count = min(self.keep_topk, int((~teacher_mask[image_idx]).sum()))
            indices = torch.topk(objectness, k=count, dim=0).indices.detach()
            p_t = teacher_probs[image_idx, indices].clamp(1e-6, 1 - 1e-6)
            p_s = student_probs[image_idx, indices].clamp(1e-6, 1 - 1e-6)
            kl = (p_t * (p_t.log() - p_s.log()) +
                  (1 - p_t) * ((1 - p_t).log() - (1 - p_s).log()))
            per_image.append(kl.mean())
            chosen += count
        loss = torch.stack(per_image).mean() if per_image else \
            student_dense.sum() * 0.0
        return loss, {
            'positions': chosen,
            'mean_teacher_probability': float(teacher_probs.mean().cpu()),
            'mean_student_probability': float(
                student_probs.detach().mean().cpu())}

    def _should_run_sgr_path(self) -> bool:
        if not self.sgr_enabled:
            return False
        if self.sgr_alpha_bg != 1.0 or self.sgr_soft_ubp_enabled:
            return True
        if (self.sgr_lambda_gfr != 0.0 or self.sgr_lambda_grr != 0.0
                or self.sgr_lambda_rank != 0.0
                or self.sgr_lambda_scr != 0.0
                or self.sgr_lambda_crsb != 0.0
                or self.sgr_lambda_rvrd != 0.0):
            return True
        return self.sgr_build_teacher_zero_effect

    def _sgr_warmup_factor(self, ref: Tensor,
                           warmup_iters: Optional[int] = None) -> Tensor:
        warmup_iters = self.sgr_warmup_iters if warmup_iters is None else \
            int(warmup_iters)
        if not self.training or warmup_iters <= 0:
            return ref.new_tensor(1.0)
        return torch.clamp(
            ref.new_tensor(float(self._sgr_iter)) /
            float(warmup_iters), 0.0, 1.0)

    def _build_sgr_teacher(self):
        if self.sgr_teacher_config is None:
            raise ValueError('sgr_cfg.teacher_config is required.')
        if self.sgr_teacher_checkpoint is None:
            raise ValueError('sgr_cfg.teacher_checkpoint is required.')
        cfg = Config.fromfile(str(self.sgr_teacher_config))
        if cfg.get('custom_imports', None):
            import_modules_from_strings(**cfg['custom_imports'])
        teacher = MODELS.build(cfg.model)
        load_checkpoint(
            teacher, str(self.sgr_teacher_checkpoint), map_location='cpu')
        for param in teacher.parameters():
            param.requires_grad_(False)
        teacher.eval()
        object.__setattr__(self, '_sgr_teacher', teacher)
        return teacher

    def _get_sgr_teacher(self, device: torch.device):
        teacher = self.__dict__.get('_sgr_teacher', None)
        if teacher is None:
            teacher = self._build_sgr_teacher()
        teacher.to(device)
        teacher.eval()
        return teacher

    def _prepare_sgr_prompts(self, model, batch_size: int):
        maps_prompts = [
            model.get_tokens_positive_and_prompts(
                list(self.sgr_prompts), True, None, None)
        ] * batch_size
        token_maps, normalized_prompts, _, _ = zip(*maps_prompts)
        if isinstance(normalized_prompts[0], list):
            raise NotImplementedError('chunked SGR prompts are unsupported.')
        text_dict = model.language_model(list(normalized_prompts))
        if model.text_feat_map is not None:
            text_dict['embedded'] = model.text_feat_map(text_dict['embedded'])
        return text_dict, list(token_maps)

    def _prepare_semantic_anchor_prompts(self, model, batch_size: int):
        maps_prompts = [
            model.get_tokens_positive_and_prompts(
                list(self.sgr_semantic_anchor_prompts), True, None, None)
        ] * batch_size
        token_maps, normalized_prompts, _, _ = zip(*maps_prompts)
        if isinstance(normalized_prompts[0], list):
            raise NotImplementedError('chunked DSCR prompts are unsupported.')
        text_dict = model.language_model(list(normalized_prompts))
        if model.text_feat_map is not None:
            text_dict['embedded'] = model.text_feat_map(text_dict['embedded'])
        return text_dict, list(token_maps)

    @torch.no_grad()
    def _forward_sgr_teacher(self, batch_inputs: Tensor,
                             batch_data_samples: SampleList) -> dict[str,
                                                                      Any]:
        teacher = self._get_sgr_teacher(batch_inputs.device)
        text_dict, token_maps = self._prepare_sgr_prompts(
            teacher, len(batch_data_samples))
        visual_feats = teacher.extract_feat(batch_inputs)
        head_inputs = teacher.forward_transformer(visual_feats, text_dict,
                                                  batch_data_samples)
        all_cls, all_boxes = teacher.bbox_head(
            head_inputs['hidden_states'],
            head_inputs['references'],
            head_inputs['memory_text'],
            head_inputs['text_token_mask'])
        return dict(
            all_cls=all_cls,
            all_boxes=all_boxes,
            hidden_states=head_inputs['hidden_states'],
            token_maps=token_maps)

    def _generic_scores(self, logits: Tensor, token_map: dict) -> Tensor:
        class_scores = convert_grounding_to_cls_scores(
            logits=logits.sigmoid()[None], positive_maps=[token_map])[0]
        return class_scores.max(dim=-1).values

    def _select_teacher_queries(self, teacher_final_boxes: Tensor,
                                teacher_prev_boxes: Tensor,
                                teacher_scores: Tensor,
                                gt_instances) -> dict[str, Tensor]:
        stability = _bbox_overlaps_xyxy(teacher_final_boxes,
                                        teacher_prev_boxes).diag()
        base_iou = _max_iou_to_gt(teacher_final_boxes, gt_instances)
        candidate_mask = (
            (teacher_scores >= self.sgr_score_thr)
            & (stability >= self.sgr_stability_thr)
            & (base_iou < self.sgr_base_iou_thr))
        candidate_idx = torch.nonzero(
            candidate_mask, as_tuple=False).flatten()
        rank_score = torch.sqrt(
            torch.clamp(teacher_scores, min=0.0) *
            torch.clamp(stability, min=0.0))
        if candidate_idx.numel() > 0:
            order = torch.argsort(rank_score[candidate_idx], descending=True)
            selected_idx = candidate_idx[order[:self.sgr_max_protected]]
        else:
            selected_idx = candidate_idx
        return dict(
            selected_idx=selected_idx,
            candidate_idx=candidate_idx,
            rank_score=rank_score,
            stability=stability,
            base_iou=base_iou)

    def _match_teacher_student(self, student_boxes: Tensor,
                               teacher_boxes: Tensor,
                               img_meta: dict) -> dict[str, Tensor]:
        device = student_boxes.device
        if student_boxes.numel() == 0 or teacher_boxes.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return dict(student_idx=empty, teacher_local_idx=empty,
                        iou=student_boxes.new_zeros((0, )))
        student_norm = _normalize_xyxy(student_boxes, img_meta)
        teacher_norm = _normalize_xyxy(teacher_boxes, img_meta)
        student_cxcywh = bbox_xyxy_to_cxcywh(student_norm)
        teacher_cxcywh = bbox_xyxy_to_cxcywh(teacher_norm)
        l1_cost = torch.cdist(student_cxcywh, teacher_cxcywh, p=1)
        giou = bbox_overlaps(student_norm, teacher_norm, mode='giou')
        cost = 2.0 * l1_cost + 5.0 * (1.0 - giou)
        rows, cols = linear_sum_assignment(cost.detach().float().cpu().numpy())
        rows_t = torch.as_tensor(rows, dtype=torch.long, device=device)
        cols_t = torch.as_tensor(cols, dtype=torch.long, device=device)
        ious = _bbox_overlaps_xyxy(student_boxes, teacher_boxes)
        matched_iou = ious[rows_t, cols_t] if rows_t.numel() else \
            student_boxes.new_zeros((0, ))
        keep = matched_iou >= self.sgr_match_iou_thr
        return dict(
            student_idx=rows_t[keep],
            teacher_local_idx=cols_t[keep],
            iou=matched_iou[keep])

    def _build_sgr_matches(self, teacher_out: dict[str, Any],
                           student_cls_scores: Tensor,
                           student_bbox_preds: Tensor,
                           student_feats: Tensor,
                           batch_gt_instances: InstanceList,
                           batch_img_metas: List[dict]) -> dict[str, Any]:
        protected_mask = torch.zeros(
            student_bbox_preds.shape[:2],
            dtype=torch.bool,
            device=student_bbox_preds.device)
        protected_weights = student_bbox_preds.new_zeros(
            student_bbox_preds.shape[:2])
        base_pos_mask = self.bbox_head.get_base_pos_mask(
            student_cls_scores.detach(), student_bbox_preds.detach(),
            batch_gt_instances, batch_img_metas)
        student_feat_rows = []
        teacher_feat_rows = []
        match_weight_rows = []
        match_batch_rows = []
        match_student_idx_rows = []
        match_teacher_box_rows = []
        match_teacher_score_rows = []
        stats = dict(
            candidate_counts=[],
            selected_counts=[],
            protected_counts=[],
            matched_base_pos_overlap=0,
            protected_base_pos_overlap=0,
            protected_dn_overlap=0,
            match_iou=[])

        for b, img_meta in enumerate(batch_img_metas):
            teacher_final_boxes = _boxes_current_xyxy(
                teacher_out['all_boxes'][-1, b].detach(), img_meta)
            teacher_prev_boxes = _boxes_current_xyxy(
                teacher_out['all_boxes'][-2, b].detach(), img_meta)
            teacher_scores = self._generic_scores(
                teacher_out['all_cls'][-1, b].detach(),
                teacher_out['token_maps'][b]).to(student_bbox_preds.device)
            selected = self._select_teacher_queries(
                teacher_final_boxes, teacher_prev_boxes, teacher_scores,
                batch_gt_instances[b])
            selected_idx = selected['selected_idx']
            stats['candidate_counts'].append(int(
                selected['candidate_idx'].numel()))
            stats['selected_counts'].append(int(selected_idx.numel()))
            if selected_idx.numel() == 0:
                stats['protected_counts'].append(0)
                continue

            student_boxes = _boxes_current_xyxy(student_bbox_preds[b],
                                                img_meta)
            teacher_boxes = teacher_final_boxes[selected_idx]
            matched = self._match_teacher_student(student_boxes, teacher_boxes,
                                                  img_meta)
            if matched['student_idx'].numel() == 0:
                stats['protected_counts'].append(0)
                continue

            s_idx = matched['student_idx']
            t_local = matched['teacher_local_idx']
            t_idx = selected_idx[t_local]
            is_base_pos = base_pos_mask[b, s_idx]
            stats['matched_base_pos_overlap'] += int(is_base_pos.sum().item())
            keep = ~is_base_pos
            s_idx = s_idx[keep]
            t_idx = t_idx[keep]
            t_local = t_local[keep]
            match_iou = matched['iou'][keep]
            if s_idx.numel() == 0:
                stats['protected_counts'].append(0)
                continue

            protected_mask[b, s_idx] = True
            teacher_conf = torch.clamp(
                (teacher_scores[t_idx] - self.sgr_score_thr) /
                max(1.0 - self.sgr_score_thr, 1e-6), min=0.0, max=1.0)
            teacher_stability = selected['stability'][t_idx].to(
                student_feats.device)
            stability_conf = torch.clamp(
                (teacher_stability - self.sgr_stability_thr) /
                max(1.0 - self.sgr_stability_thr, 1e-6),
                min=0.0,
                max=1.0)
            soft_protect = torch.sqrt(teacher_conf * stability_conf)
            protected_weights[b, s_idx] = soft_protect.detach()
            student_feat_rows.append(student_feats[b, s_idx])
            teacher_feat_rows.append(teacher_out['hidden_states'][-1, b,
                                                                  t_idx])
            match_weight_rows.append(selected['rank_score'][selected_idx[
                t_local]].detach().to(student_feats.device))
            match_batch_rows.append(
                torch.full_like(s_idx, int(b), dtype=torch.long))
            match_student_idx_rows.append(s_idx.detach())
            match_teacher_box_rows.append(teacher_boxes[t_local].detach())
            match_teacher_score_rows.append(
                teacher_scores[t_idx].detach().to(student_feats.device))
            stats['protected_counts'].append(int(s_idx.numel()))
            stats['match_iou'].extend([
                float(x) for x in match_iou.detach().float().cpu().tolist()
            ])
            stats.setdefault('soft_protect_weights', []).extend([
                float(x) for x in soft_protect.detach().float().cpu().tolist()
            ])

        stats['protected_base_pos_overlap'] = int(
            (protected_mask & base_pos_mask).sum().detach().item())
        if len(student_feat_rows) > 0:
            student_feats_matched = torch.cat(student_feat_rows, dim=0)
            teacher_feats_matched = torch.cat(teacher_feat_rows, dim=0)
            match_weights = torch.cat(match_weight_rows, dim=0)
            match_batch_idx = torch.cat(match_batch_rows, dim=0)
            match_student_idx = torch.cat(match_student_idx_rows, dim=0)
            match_teacher_boxes = torch.cat(match_teacher_box_rows, dim=0)
            match_teacher_scores = torch.cat(match_teacher_score_rows, dim=0)
        else:
            student_feats_matched = student_feats.new_zeros(
                (0, student_feats.size(-1)))
            teacher_feats_matched = student_feats.new_zeros(
                (0, student_feats.size(-1)))
            match_weights = student_feats.new_zeros((0, ))
            match_batch_idx = torch.empty(
                0, dtype=torch.long, device=student_feats.device)
            match_student_idx = torch.empty(
                0, dtype=torch.long, device=student_feats.device)
            match_teacher_boxes = student_feats.new_zeros((0, 4))
            match_teacher_scores = student_feats.new_zeros((0, ))
        return dict(
            protected_mask=protected_mask,
            protected_weights=protected_weights,
            base_pos_mask=base_pos_mask,
            student_feats=student_feats_matched,
            teacher_feats=teacher_feats_matched,
            match_weights=match_weights,
            match_batch_idx=match_batch_idx,
            match_student_idx=match_student_idx,
            match_teacher_boxes=match_teacher_boxes,
            match_teacher_scores=match_teacher_scores,
            stats=stats)

    def _audit_protected_layer_correspondence(
            self, all_boxes: Tensor, num_dn: int, sgr_matches: dict[str, Any],
            batch_img_metas: List[dict]) -> list[dict[str, float]]:
        """Measure whether final-layer protected query ids track one region.

        The protected set is selected at the final decoder layer.  This audit
        follows each selected student query id through every decoder layer and
        compares it with the final teacher box used by the matcher.
        """
        batch_idx = sgr_matches['match_batch_idx']
        student_idx = sgr_matches['match_student_idx']
        teacher_boxes = sgr_matches['match_teacher_boxes']
        rows: list[dict[str, float]] = []
        if batch_idx.numel() == 0:
            return rows
        for layer_idx in range(int(all_boxes.size(0))):
            ious = []
            for b, img_meta in enumerate(batch_img_metas):
                keep = batch_idx == b
                if not bool(keep.any()):
                    continue
                query_idx = student_idx[keep] + int(num_dn)
                layer_boxes = _boxes_current_xyxy(
                    all_boxes[layer_idx, b, query_idx].detach(), img_meta)
                target_boxes = teacher_boxes[keep]
                ious.append(_bbox_overlaps_xyxy(
                    layer_boxes, target_boxes).diagonal())
            values = torch.cat(ious).float()
            rows.append(
                dict(
                    decoder_layer=float(layer_idx),
                    count=float(values.numel()),
                    iou_mean=float(values.mean().cpu()),
                    iou_median=float(values.median().cpu()),
                    iou_ge_match_thr=float(
                        (values >= self.sgr_match_iou_thr).float().mean().cpu()),
                    iou_ge_0_5=float((values >= 0.5).float().mean().cpu())))
        return rows

    def _loss_gfr(self, student_feats: Tensor, teacher_feats: Tensor,
                  weights: Tensor, ref: Tensor) -> Tensor:
        if student_feats.numel() == 0:
            return ref.sum() * 0.0
        cos = F.cosine_similarity(student_feats, teacher_feats.detach(),
                                  dim=-1)
        loss = 1.0 - cos
        weights = weights.detach().to(device=loss.device, dtype=loss.dtype)
        return (loss * weights).sum() / max(int(loss.numel()), 1)

    def _get_generic_text_cache(self, device: torch.device) -> dict[str, Any]:
        cache = self._sgr_generic_text_cache
        if cache is None or cache['embedded'].device != device:
            with torch.no_grad():
                text_dict, token_maps = self._prepare_sgr_prompts(self, 1)
            cache = dict(
                embedded=text_dict['embedded'].detach().to(device),
                text_token_mask=text_dict['text_token_mask'].detach().to(
                    device),
                token_map=token_maps[0])
            self._sgr_generic_text_cache = cache
        return cache

    def _get_semantic_anchor_text_cache(self, model, device: torch.device,
                                        cache_name: str) -> dict[str, Any]:
        cache = getattr(self, cache_name)
        if cache is None or cache['embedded'].device != device:
            with torch.no_grad():
                text_dict, token_maps = self._prepare_semantic_anchor_prompts(
                    model, 1)
            cache = dict(
                embedded=text_dict['embedded'].detach().to(device),
                text_token_mask=text_dict['text_token_mask'].detach().to(
                    device),
                token_map=token_maps[0])
            setattr(self, cache_name, cache)
        return cache

    def _generic_saliency(self, student_feats: Tensor) -> Tensor:
        cache = self._get_generic_text_cache(student_feats.device)
        bs = student_feats.size(0)
        generic_text = cache['embedded'].expand(bs, -1, -1)
        generic_mask = cache['text_token_mask'].expand(bs, -1)
        logits = self.bbox_head.forward_cls(-1, student_feats, generic_text,
                                            generic_mask)
        prompt_logits = convert_grounding_to_cls_scores(
            logits=logits, positive_maps=[cache['token_map']] * bs)
        temp = max(self.sgr_generic_temperature, 1e-6)
        return temp * torch.logsumexp(prompt_logits / temp, dim=-1)

    def _generic_objectness(self, student_feats: Tensor) -> Tensor:
        """Return differentiable D1 generic objectness in [0, 1]."""
        cache = self._get_generic_text_cache(student_feats.device)
        bs = student_feats.size(0)
        generic_text = cache['embedded'].expand(bs, -1, -1)
        generic_mask = cache['text_token_mask'].expand(bs, -1)
        logits = self.bbox_head.forward_cls(-1, student_feats, generic_text,
                                            generic_mask)
        prompt_scores = convert_grounding_to_cls_scores(
            logits=logits.sigmoid(),
            positive_maps=[cache['token_map']] * bs)
        return prompt_scores.max(dim=-1).values

    def _loss_rank_retention(self, student_feats: Tensor,
                             protected_mask: Tensor,
                             protected_weights: Tensor,
                             base_pos_mask: Tensor) -> Tensor:
        """Rank protected queries above the hardest generic backgrounds."""
        if protected_mask.sum() == 0:
            return student_feats.sum() * 0.0
        objectness = self._generic_objectness(student_feats)
        numerator = objectness.sum() * 0.0
        denominator = objectness.new_tensor(0.0)
        for b in range(student_feats.size(0)):
            pos_idx = torch.nonzero(
                protected_mask[b], as_tuple=False).flatten()
            bg_idx = torch.nonzero(
                ~protected_mask[b] & ~base_pos_mask[b],
                as_tuple=False).flatten()
            if pos_idx.numel() == 0 or bg_idx.numel() == 0:
                continue
            num_neg = min(max(self.sgr_rank_negatives, 1),
                          int(bg_idx.numel()))
            hard_order = torch.topk(
                objectness[b, bg_idx].detach(), k=num_neg).indices
            neg_idx = bg_idx[hard_order]
            pos_scores = objectness[b, pos_idx].unsqueeze(1)
            neg_scores = objectness[b, neg_idx].unsqueeze(0)
            margin = objectness.new_tensor(self.sgr_rank_margin)
            per_pos = F.relu(margin - pos_scores + neg_scores).mean(dim=1)
            weights = protected_weights[b, pos_idx].detach().to(
                device=per_pos.device, dtype=per_pos.dtype)
            numerator = numerator + (per_pos * weights).sum()
            denominator = denominator + weights.sum()
        if bool((denominator <= 0).detach()):
            return student_feats.sum() * 0.0
        return numerator / denominator.clamp_min(1e-6)

    def _loss_grr(self, student_feats: Tensor, protected_mask: Tensor,
                  base_pos_mask: Tensor, matching_cls_scores: Tensor,
                  text_token_mask: Tensor) -> Tensor:
        if protected_mask.sum() == 0:
            return student_feats.sum() * 0.0
        saliency = self._generic_saliency(student_feats)
        text_masks = text_token_mask.new_zeros(
            (text_token_mask.size(0), self.bbox_head.max_text_len))
        text_masks[:, :text_token_mask.size(1)] = text_token_mask
        valid_text = text_masks[:, :matching_cls_scores.size(-1)].bool()
        base_scores = matching_cls_scores.detach().sigmoid().masked_fill(
            ~valid_text[:, None, :], -1.0).max(dim=-1).values
        losses = []
        for b in range(student_feats.size(0)):
            pos_idx = torch.nonzero(
                protected_mask[b], as_tuple=False).flatten()
            bg_mask = ~protected_mask[b] & ~base_pos_mask[b]
            bg_idx = torch.nonzero(bg_mask, as_tuple=False).flatten()
            if pos_idx.numel() == 0 or bg_idx.numel() == 0:
                continue
            bg_order = torch.argsort(base_scores[b, bg_idx], descending=False)
            bg_idx = bg_idx[bg_order]
            need = int(pos_idx.numel()) * max(self.sgr_grr_negatives, 1)
            repeats = math.ceil(need / max(int(bg_idx.numel()), 1))
            neg_idx = bg_idx.repeat(repeats)[:need].view(
                int(pos_idx.numel()), max(self.sgr_grr_negatives, 1))
            pos_score = saliency[b, pos_idx].unsqueeze(1)
            neg_score = saliency[b, neg_idx]
            margin = saliency.new_tensor(self.sgr_grr_margin)
            temp = max(self.sgr_grr_temperature, 1e-6)
            losses.append(
                F.softplus(-(pos_score - neg_score - margin) / temp).mean())
        if len(losses) == 0:
            return student_feats.sum() * 0.0
        return sum(losses) / len(losses)

    def _semantic_anchor_scores(self, feats: Tensor, model,
                                cache: dict[str, Any]) -> Tensor:
        if feats.numel() == 0:
            return feats.new_zeros(
                (0, len(self.sgr_semantic_anchor_prompts)))
        anchor_text = cache['embedded'].to(feats.device)
        anchor_mask = cache['text_token_mask'].to(feats.device)
        logits = model.bbox_head.forward_cls(
            -1, feats.unsqueeze(0), anchor_text, anchor_mask)
        scores = convert_grounding_to_cls_scores(
            logits=logits, positive_maps=[cache['token_map']])[0]
        return scores.float()

    def _safe_corrcoef(self, x: Tensor, y: Tensor, ref: Tensor) -> Tensor:
        if x.numel() < 2 or y.numel() < 2:
            return ref.new_tensor(0.0)
        x = x.float() - x.float().mean()
        y = y.float() - y.float().mean()
        denom = x.norm() * y.norm()
        if bool(denom.detach() <= self.sgr_scr_eps):
            return ref.new_tensor(0.0)
        return (x * y).sum() / denom.clamp_min(self.sgr_scr_eps)

    def _scr_kl_components(
            self, teacher_scores: Tensor,
            student_scores: Tensor) -> dict[str, Tensor]:
        """Return the legacy full KL and the exact base-collapsed split.

        ``full_legacy`` deliberately preserves the historical epsilon-based
        implementation so ``scr_target_mode='full'`` is bit-for-bit
        compatible.  The collapsed decomposition uses log-softmax and
        logsumexp, matching the probability grouping in the experiment plan.
        """
        temp = max(self.sgr_scr_temperature, self.sgr_scr_eps)
        eps = self.sgr_scr_eps
        teacher_log = F.log_softmax(teacher_scores.float() / temp, dim=-1)
        student_log = F.log_softmax(student_scores.float() / temp, dim=-1)
        # Keep these two softmax calls identical to the historical path.  The
        # log-softmax tensors below are only used by the exact decomposition.
        teacher_probs = F.softmax(
            teacher_scores.float() / temp, dim=-1).detach()
        student_probs = F.softmax(student_scores.float() / temp, dim=-1)
        full_legacy = (
            teacher_probs *
            ((teacher_probs + eps).log() -
             (student_probs + eps).log())).sum(dim=-1)
        full_exact = (
            teacher_probs * (teacher_log.detach() - student_log)).sum(dim=-1)

        base_count = int(self.sgr_semantic_base_count)
        anchor_count = int(teacher_scores.size(-1))
        if not (0 < base_count < anchor_count):
            raise ValueError(
                'base_collapsed SCR requires at least one base and one '
                f'generic anchor, got base={base_count}, total={anchor_count}')
        teacher_base_log = torch.logsumexp(
            teacher_log[:, :base_count], dim=-1, keepdim=True)
        student_base_log = torch.logsumexp(
            student_log[:, :base_count], dim=-1, keepdim=True)
        teacher_coarse_log = torch.cat(
            [teacher_base_log, teacher_log[:, base_count:]], dim=-1).detach()
        student_coarse_log = torch.cat(
            [student_base_log, student_log[:, base_count:]], dim=-1)
        teacher_coarse_probs = teacher_coarse_log.exp()
        coarse = (
            teacher_coarse_probs *
            (teacher_coarse_log - student_coarse_log)).sum(dim=-1)

        teacher_cond_log = (
            teacher_log[:, :base_count] - teacher_base_log).detach()
        student_cond_log = (
            student_log[:, :base_count] - student_base_log)
        teacher_base_mass = teacher_base_log.detach().exp().squeeze(-1)
        within = teacher_base_mass * (
            teacher_cond_log.exp() *
            (teacher_cond_log - student_cond_log)).sum(dim=-1)
        decomposition_error = (full_exact - coarse - within).abs()
        return dict(
            teacher_probs=teacher_probs,
            student_probs=student_probs,
            full_legacy=full_legacy,
            full_exact=full_exact,
            coarse=coarse,
            base_within=within,
            decomposition_error=decomposition_error)

    def _loss_scr(self, student_feats: Tensor, teacher_feats: Tensor,
                  ref: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        zero = ref.sum() * 0.0
        empty_stats = dict(
            scr_kl=ref.new_tensor(0.0),
            scr_kl_full=ref.new_tensor(0.0),
            scr_kl_full_exact=ref.new_tensor(0.0),
            scr_kl_coarse=ref.new_tensor(0.0),
            scr_kl_base_within=ref.new_tensor(0.0),
            scr_kl_decomposition_error=ref.new_tensor(0.0),
            scr_kl_legacy_exact_gap=ref.new_tensor(0.0),
            scr_js=ref.new_tensor(0.0),
            scr_entropy_teacher=ref.new_tensor(0.0),
            scr_entropy_student=ref.new_tensor(0.0),
            scr_base_growth=ref.new_tensor(0.0),
            scr_base_growth_q90=ref.new_tensor(0.0),
            scr_js_base_growth_corr=ref.new_tensor(0.0),
            scr_weight_mean=ref.new_tensor(0.0),
            scr_weight_max=ref.new_tensor(0.0),
            scr_feature_grad_norm_raw=ref.new_tensor(0.0),
            scr_feature_grad_rms_raw=ref.new_tensor(0.0),
            scr_feature_grad_norm_full_raw=ref.new_tensor(0.0),
            scr_feature_grad_norm_coarse_raw=ref.new_tensor(0.0),
            scr_feature_grad_cosine_full_coarse=ref.new_tensor(0.0),
            scr_anchor_count=ref.new_tensor(
                float(len(self.sgr_semantic_anchor_prompts))))
        if student_feats.numel() == 0 or not self.sgr_semantic_anchor_prompts:
            self._last_scr_audit_tensors = {}
            return zero, empty_stats

        device = student_feats.device
        teacher = self._get_sgr_teacher(device)
        student_cache = self._get_semantic_anchor_text_cache(
            self, device, '_sgr_anchor_text_cache')
        teacher_cache = self._get_semantic_anchor_text_cache(
            teacher, device, '_sgr_teacher_anchor_text_cache')
        eps = self.sgr_scr_eps

        with torch.no_grad():
            teacher_scores = self._semantic_anchor_scores(
                teacher_feats.detach(), teacher, teacher_cache)
        student_scores = self._semantic_anchor_scores(
            student_feats, self, student_cache)
        need_decomposition = (
            self.sgr_scr_target_mode == 'base_collapsed' or
            self.sgr_scr_log_decomposition or self.sgr_scr_audit_tensors or
            self.sgr_scr_direction_audit_tensors)
        if need_decomposition:
            terms = self._scr_kl_components(teacher_scores, student_scores)
        else:
            # Preserve the pre-existing full-SCR path exactly when the new
            # target and its audit are disabled.
            temp = max(self.sgr_scr_temperature, self.sgr_scr_eps)
            teacher_probs = F.softmax(
                teacher_scores / temp, dim=-1).detach()
            student_probs_legacy = F.softmax(
                student_scores.float() / temp, dim=-1)
            full_legacy = (
                teacher_probs *
                ((teacher_probs + eps).log() -
                 (student_probs_legacy + eps).log())).sum(dim=-1)
            terms = dict(
                teacher_probs=teacher_probs,
                student_probs=student_probs_legacy,
                full_legacy=full_legacy,
                full_exact=full_legacy,
                coarse=torch.zeros_like(full_legacy),
                base_within=torch.zeros_like(full_legacy),
                decomposition_error=torch.zeros_like(full_legacy))
        teacher_probs_det = terms['teacher_probs'].detach()
        student_probs = terms['student_probs']
        student_probs_det = student_probs.detach()
        kl_full = terms['full_legacy']
        kl_coarse = terms['coarse']
        kl = kl_full if self.sgr_scr_target_mode == 'full' else kl_coarse
        mixture = 0.5 * (teacher_probs_det + student_probs_det)
        js = 0.5 * (
            teacher_probs_det *
            ((teacher_probs_det + eps).log() -
             (mixture + eps).log())).sum(dim=-1)
        js = js + 0.5 * (
            student_probs_det *
            ((student_probs_det + eps).log() -
             (mixture + eps).log())).sum(dim=-1)
        if self.sgr_scr_weight_mode == 'uniform':
            drift_weights = torch.ones_like(js.detach())
        else:
            drift_weights = (
                js.detach() / js.detach().mean().clamp_min(eps)).clamp(
                    min=self.sgr_scr_weight_min, max=self.sgr_scr_weight_max)
        loss = (kl * drift_weights).sum() / max(int(kl.numel()), 1)
        loss_full = (kl_full * drift_weights).sum() / max(
            int(kl_full.numel()), 1)
        loss_coarse = (kl_coarse * drift_weights).sum() / max(
            int(kl_coarse.numel()), 1)

        # Exact SCR-only gradient strength at the selected decoder features.
        # This is recorded before lambda/warmup so weighted and uniform SCR can
        # be compared without conflating the weighting rule with its schedule.
        # autograd stops at student_feats, so this does not traverse the full
        # detector twice and does not modify parameter gradients.
        feature_grad = torch.autograd.grad(
            loss, student_feats, retain_graph=True, allow_unused=True)[0]
        if feature_grad is None:
            feature_grad_norm = ref.new_tensor(0.0)
            feature_grad_rms = ref.new_tensor(0.0)
        else:
            feature_grad_float = feature_grad.detach().float()
            feature_grad_norm = feature_grad_float.norm()
            feature_grad_rms = feature_grad_float.square().mean().sqrt()

        full_grad_norm = feature_grad_norm
        coarse_grad_norm = feature_grad_norm
        grad_cosine = ref.new_tensor(1.0)
        if self.sgr_scr_log_decomposition:
            full_grad = torch.autograd.grad(
                loss_full,
                student_feats,
                retain_graph=True,
                allow_unused=True)[0]
            coarse_grad = torch.autograd.grad(
                loss_coarse,
                student_feats,
                retain_graph=True,
                allow_unused=True)[0]
            if full_grad is None:
                full_grad_norm = ref.new_tensor(0.0)
            else:
                full_grad_norm = full_grad.detach().float().norm()
            if coarse_grad is None:
                coarse_grad_norm = ref.new_tensor(0.0)
            else:
                coarse_grad_norm = coarse_grad.detach().float().norm()
            if (full_grad is None or coarse_grad is None or
                    float(full_grad_norm) == 0.0 or
                    float(coarse_grad_norm) == 0.0):
                grad_cosine = ref.new_tensor(0.0)
            else:
                grad_cosine = F.cosine_similarity(
                    full_grad.detach().float().flatten(),
                    coarse_grad.detach().float().flatten(),
                    dim=0)
        if (self.sgr_scr_audit_tensors or
                self.sgr_scr_direction_audit_tensors):
            self._last_scr_audit_tensors = dict(
                loss_full=loss_full,
                loss_coarse=loss_coarse,
                student_feats=student_feats)
        else:
            self._last_scr_audit_tensors = {}

        teacher_entropy = -(
            teacher_probs_det *
            (teacher_probs_det + eps).log()).sum(dim=-1)
        student_entropy = -(
            student_probs_det *
            (student_probs_det + eps).log()).sum(dim=-1)
        base_count = min(self.sgr_semantic_base_count,
                         teacher_scores.size(-1))
        teacher_max_base = teacher_scores[:, :base_count].max(dim=-1).values
        student_max_base = student_scores.detach()[:, :base_count].max(
            dim=-1).values
        base_growth = student_max_base.float() - teacher_max_base.float()
        stats = dict(
            scr_kl=kl.detach().mean(),
            scr_kl_full=kl_full.detach().mean(),
            scr_kl_full_exact=terms['full_exact'].detach().mean(),
            scr_kl_coarse=kl_coarse.detach().mean(),
            scr_kl_base_within=terms['base_within'].detach().mean(),
            scr_kl_decomposition_error=(
                terms['decomposition_error'].detach().max()),
            scr_kl_legacy_exact_gap=(
                kl_full.detach() - terms['full_exact'].detach()).abs().max(),
            scr_js=js.detach().mean(),
            scr_entropy_teacher=teacher_entropy.detach().mean(),
            scr_entropy_student=student_entropy.detach().mean(),
            scr_base_growth=base_growth.detach().mean(),
            scr_base_growth_q90=torch.quantile(
                base_growth.detach().float(), 0.9),
            scr_js_base_growth_corr=self._safe_corrcoef(
                js.detach(), base_growth.detach(), ref).detach(),
            scr_weight_mean=drift_weights.detach().mean(),
            scr_weight_max=drift_weights.detach().max(),
            scr_feature_grad_norm_raw=feature_grad_norm,
            scr_feature_grad_rms_raw=feature_grad_rms,
            scr_feature_grad_norm_full_raw=full_grad_norm,
            scr_feature_grad_norm_coarse_raw=coarse_grad_norm,
            scr_feature_grad_cosine_full_coarse=grad_cosine,
            scr_anchor_count=ref.new_tensor(
                float(len(self.sgr_semantic_anchor_prompts))))
        return loss, stats

    def _get_crsb_cache(self) -> Optional[dict[str, Any]]:
        if self.sgr_crsb_cache_file is None:
            return None
        cache = self._sgr_crsb_cache
        if cache is not None:
            return cache
        payload = torch.load(str(self.sgr_crsb_cache_file), map_location='cpu')
        anchors = tuple(payload.get('anchors', ()))
        if anchors and anchors != self.sgr_semantic_anchor_prompts:
            raise ValueError('CRSB cache anchor order does not match config: '
                             f'{anchors} != {self.sgr_semantic_anchor_prompts}')
        raw_by_image = payload.get('by_image', {})
        by_image = {}
        for image_id, row in raw_by_image.items():
            by_image[int(image_id)] = dict(
                boxes=torch.as_tensor(row['boxes'], dtype=torch.float32),
                tight_logits=torch.as_tensor(
                    row['tight_logits'], dtype=torch.float32),
                context_logits=torch.as_tensor(
                    row.get('context_logits', row['tight_logits']),
                    dtype=torch.float32),
                tight_context_cosine=torch.as_tensor(
                    row['tight_context_cosine'], dtype=torch.float32),
                generic_scores=torch.as_tensor(
                    row.get('generic_scores',
                            torch.zeros(len(row['boxes']))),
                    dtype=torch.float32),
                query_indices=torch.as_tensor(
                    row.get('query_indices',
                            torch.arange(len(row['boxes']))),
                    dtype=torch.long))
        cache = dict(payload)
        cache['by_image'] = by_image
        self._sgr_crsb_cache = cache
        return cache

    def _match_crsb_cache(self, sgr_matches: dict[str, Any],
                          batch_img_metas: List[dict],
                          ref: Tensor) -> dict[str, Tensor]:
        n = int(sgr_matches['student_feats'].size(0))
        anchor_count = len(self.sgr_semantic_anchor_prompts)
        device = ref.device
        out = dict(
            valid=torch.zeros(n, dtype=torch.bool, device=device),
            tight_logits=ref.new_zeros((n, anchor_count)),
            context_logits=ref.new_zeros((n, anchor_count)),
            rho=ref.new_zeros((n, )),
            match_iou=ref.new_zeros((n, )),
            cache_count=ref.new_tensor(0.0),
            image_count=ref.new_tensor(float(len(batch_img_metas))))
        if n == 0:
            return out
        cache = self._get_crsb_cache()
        if cache is None:
            return out

        batch_idx = sgr_matches['match_batch_idx']
        teacher_boxes = sgr_matches['match_teacher_boxes']
        cache_count_total = 0
        for b, img_meta in enumerate(batch_img_metas):
            row_idx = torch.nonzero(
                batch_idx == int(b), as_tuple=False).flatten()
            if row_idx.numel() == 0:
                continue
            image_id = int(img_meta.get('img_id', img_meta.get('image_id')))
            cache_row = cache['by_image'].get(image_id, None)
            if cache_row is None or cache_row['boxes'].numel() == 0:
                continue
            cache_boxes = cache_row['boxes'].to(device=device,
                                                dtype=ref.dtype)
            cache_boxes = _boxes_original_to_current(cache_boxes, img_meta)
            cache_count_total += int(cache_boxes.size(0))
            ious = _bbox_overlaps_xyxy(teacher_boxes[row_idx].to(ref.dtype),
                                       cache_boxes)
            if ious.numel() == 0:
                continue
            best_iou, best_col = ious.max(dim=1)
            keep = best_iou >= self.sgr_crsb_cache_match_iou
            if not bool(keep.any()):
                continue
            rows = row_idx[keep]
            cols = best_col[keep].detach().cpu()
            out['valid'][rows] = True
            out['match_iou'][rows] = best_iou[keep].detach()
            out['tight_logits'][rows] = cache_row['tight_logits'][
                cols].to(device=device, dtype=ref.dtype)
            out['context_logits'][rows] = cache_row['context_logits'][
                cols].to(device=device, dtype=ref.dtype)
            out['rho'][rows] = cache_row['tight_context_cosine'][
                cols].to(device=device, dtype=ref.dtype).clamp(0.0, 1.0)
        out['cache_count'] = ref.new_tensor(float(cache_count_total))
        return out

    def _confidence_from_probs(self, probs: Tensor) -> Tensor:
        eps = self.sgr_scr_eps
        entropy = -(probs * (probs + eps).log()).sum(dim=-1)
        denom = math.log(max(int(probs.size(-1)), 2))
        return (1.0 - entropy / denom).clamp(0.0, 1.0)

    def _quantile_or_zero(self, values: Tensor, q: float,
                          ref: Tensor) -> Tensor:
        if values.numel() == 0:
            return ref.new_tensor(0.0)
        return torch.quantile(values.detach().float(), q)

    def _peak_base_guard(self, p_mix: Tensor, p_g: Tensor) -> tuple[
            Tensor, dict[str, Tensor]]:
        if not self.sgr_peak_base_guard_enabled:
            zero = p_mix.new_zeros((p_mix.size(0), ))
            max_base_before = p_mix[:, :min(self.sgr_semantic_base_count,
                                            p_mix.size(-1))].max(
                                                dim=-1).values
            return p_mix, dict(
                active=zero.bool(),
                clipped=zero,
                max_base_before=max_base_before,
                max_base_after=max_base_before,
                generic_mass_before=p_mix[:, min(
                    self.sgr_semantic_base_count, p_mix.size(-1)):].sum(
                        dim=-1),
                generic_mass_after=p_mix[:, min(
                    self.sgr_semantic_base_count, p_mix.size(-1)):].sum(
                        dim=-1))
        base_count = min(self.sgr_semantic_base_count, p_mix.size(-1))
        if base_count <= 0 or base_count >= p_mix.size(-1):
            zero = p_mix.new_zeros((p_mix.size(0), ))
            return p_mix, dict(
                active=zero.bool(),
                clipped=zero,
                max_base_before=zero,
                max_base_after=zero,
                generic_mass_before=zero,
                generic_mass_after=zero)
        p_t = p_mix.clone()
        max_base_before, top_base = p_mix[:, :base_count].max(dim=-1)
        cap = (p_g[:, :base_count].max(dim=-1).values +
               self.sgr_crsb_delta_peak).clamp(max=1.0)
        active = max_base_before > cap
        clipped = (max_base_before - cap).clamp_min(0.0)
        if bool(active.any()):
            rows = torch.arange(p_t.size(0), device=p_t.device)[active]
            cols = top_base[active]
            p_t[rows, cols] = cap[active]
            gen = p_g[active, base_count:]
            gen_mass = gen.sum(dim=-1, keepdim=True)
            uniform = torch.full_like(gen, 1.0 / gen.size(1))
            alloc = torch.where(gen_mass > self.sgr_scr_eps,
                                gen / gen_mass.clamp_min(self.sgr_scr_eps),
                                uniform)
            p_t[active, base_count:] = p_t[active, base_count:] + clipped[
                active].unsqueeze(1) * alloc
        p_t = p_t / p_t.sum(dim=-1, keepdim=True).clamp_min(self.sgr_scr_eps)
        max_base_after = p_t[:, :base_count].max(dim=-1).values
        return p_t, dict(
            active=active,
            clipped=clipped,
            max_base_before=max_base_before,
            max_base_after=max_base_after,
            generic_mass_before=p_mix[:, base_count:].sum(dim=-1),
            generic_mass_after=p_t[:, base_count:].sum(dim=-1))

    def _loss_crsb(self, student_feats: Tensor, teacher_feats: Tensor,
                   crsb_cache: dict[str, Tensor],
                   ref: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if (self.sgr_crsb_beta_max == 0.0
                and self.sgr_crsb_guard_bypass_when_beta_zero):
            loss, scr_stats = self._loss_scr(student_feats, teacher_feats, ref)
            stats = {
                'crsb_' + key[4:]: value
                for key, value in scr_stats.items() if key.startswith('scr_')
            }
            stats.update(
                crsb_cache_hit_rate=ref.new_tensor(0.0),
                crsb_valid_per_img=ref.new_tensor(0.0),
                crsb_beta_mean=ref.new_tensor(0.0),
                crsb_beta_q50=ref.new_tensor(0.0),
                crsb_beta_q95=ref.new_tensor(0.0),
                crsb_guard_active_rate=ref.new_tensor(0.0),
                crsb_clipped_mass_mean=ref.new_tensor(0.0),
                crsb_clipped_mass_q95=ref.new_tensor(0.0))
            return loss, stats

        zero = ref.sum() * 0.0
        empty_stats = dict(
            crsb_kl=ref.new_tensor(0.0),
            crsb_js=ref.new_tensor(0.0),
            crsb_entropy_generic=ref.new_tensor(0.0),
            crsb_entropy_remote=ref.new_tensor(0.0),
            crsb_entropy_student=ref.new_tensor(0.0),
            crsb_cache_hit_rate=ref.new_tensor(0.0),
            crsb_valid_per_img=ref.new_tensor(0.0),
            crsb_beta_mean=ref.new_tensor(0.0),
            crsb_beta_q50=ref.new_tensor(0.0),
            crsb_beta_q95=ref.new_tensor(0.0),
            crsb_guard_active_rate=ref.new_tensor(0.0),
            crsb_clipped_mass_mean=ref.new_tensor(0.0),
            crsb_clipped_mass_q95=ref.new_tensor(0.0),
            crsb_max_base_before=ref.new_tensor(0.0),
            crsb_max_base_after=ref.new_tensor(0.0),
            crsb_generic_mass_before=ref.new_tensor(0.0),
            crsb_generic_mass_after=ref.new_tensor(0.0),
            crsb_remote_confidence=ref.new_tensor(0.0),
            crsb_rho=ref.new_tensor(0.0),
            crsb_weight_mean=ref.new_tensor(0.0),
            crsb_weight_max=ref.new_tensor(0.0),
            crsb_anchor_count=ref.new_tensor(
                float(len(self.sgr_semantic_anchor_prompts))))
        if student_feats.numel() == 0 or not self.sgr_semantic_anchor_prompts:
            return zero, empty_stats

        device = student_feats.device
        teacher = self._get_sgr_teacher(device)
        student_cache = self._get_semantic_anchor_text_cache(
            self, device, '_sgr_anchor_text_cache')
        teacher_cache = self._get_semantic_anchor_text_cache(
            teacher, device, '_sgr_teacher_anchor_text_cache')
        eps = self.sgr_scr_eps
        tau_g = max(self.sgr_crsb_tau_g, eps)
        tau_r = max(self.sgr_crsb_tau_r, eps)
        with torch.no_grad():
            teacher_scores = self._semantic_anchor_scores(
                teacher_feats.detach(), teacher, teacher_cache)
            p_g = F.softmax(teacher_scores / tau_g, dim=-1)
            remote_logits = crsb_cache['tight_logits'].to(
                device=device, dtype=teacher_scores.dtype)
            p_r = F.softmax(remote_logits / tau_r, dim=-1)
            valid = crsb_cache['valid'].to(device=device)
            rho = crsb_cache['rho'].to(device=device,
                                       dtype=teacher_scores.dtype)
            remote_conf = self._confidence_from_probs(p_r)
            beta = (self.sgr_crsb_beta_max * rho *
                    remote_conf).clamp(0.0, 1.0)
            beta = torch.where(valid, beta, beta.new_zeros(beta.shape))
            p_mix = (1.0 - beta[:, None]) * p_g + beta[:, None] * p_r
            p_mix = p_mix / p_mix.sum(dim=-1,
                                      keepdim=True).clamp_min(eps)
            p_t, guard_stats = self._peak_base_guard(p_mix, p_g)

        student_scores = self._semantic_anchor_scores(
            student_feats, self, student_cache)
        p_s = F.softmax(student_scores.float() / tau_g, dim=-1)
        p_t_det = p_t.detach()
        p_s_det = p_s.detach()
        kl = (
            p_t_det *
            ((p_t_det + eps).log() -
             (p_s.float() + eps).log())).sum(dim=-1)
        mixture = 0.5 * (p_t_det + p_s_det)
        js = 0.5 * (
            p_t_det *
            ((p_t_det + eps).log() -
             (mixture + eps).log())).sum(dim=-1)
        js = js + 0.5 * (
            p_s_det *
            ((p_s_det + eps).log() -
             (mixture + eps).log())).sum(dim=-1)
        drift_weights = (
            js.detach() / js.detach().mean().clamp_min(eps)).clamp(
                min=self.sgr_scr_weight_min, max=self.sgr_scr_weight_max)
        loss = (kl * drift_weights).sum() / max(int(kl.numel()), 1)

        p_g_det = p_g.detach()
        p_r_det = p_r.detach()
        entropy_g = -(p_g_det * (p_g_det + eps).log()).sum(dim=-1)
        entropy_r = -(p_r_det * (p_r_det + eps).log()).sum(dim=-1)
        entropy_s = -(p_s_det * (p_s_det + eps).log()).sum(dim=-1)
        valid_float = valid.detach().float()
        image_count = crsb_cache['image_count'].to(device=device,
                                                   dtype=student_feats.dtype)
        stats = dict(
            crsb_kl=kl.detach().mean(),
            crsb_js=js.detach().mean(),
            crsb_entropy_generic=entropy_g.detach().mean(),
            crsb_entropy_remote=entropy_r.detach().mean(),
            crsb_entropy_student=entropy_s.detach().mean(),
            crsb_cache_hit_rate=valid_float.mean(),
            crsb_valid_per_img=valid_float.sum() /
            image_count.clamp_min(1.0),
            crsb_beta_mean=beta.detach().float().mean(),
            crsb_beta_q50=self._quantile_or_zero(beta, 0.50, ref),
            crsb_beta_q95=self._quantile_or_zero(beta, 0.95, ref),
            crsb_guard_active_rate=guard_stats['active'].detach().float(
            ).mean(),
            crsb_clipped_mass_mean=guard_stats['clipped'].detach().float(
            ).mean(),
            crsb_clipped_mass_q95=self._quantile_or_zero(
                guard_stats['clipped'], 0.95, ref),
            crsb_max_base_before=guard_stats['max_base_before'].detach(
            ).float().mean(),
            crsb_max_base_after=guard_stats['max_base_after'].detach().float(
            ).mean(),
            crsb_generic_mass_before=guard_stats[
                'generic_mass_before'].detach().float().mean(),
            crsb_generic_mass_after=guard_stats[
                'generic_mass_after'].detach().float().mean(),
            crsb_remote_confidence=remote_conf.detach().float().mean(),
            crsb_rho=rho.detach().float().mean(),
            crsb_weight_mean=drift_weights.detach().mean(),
            crsb_weight_max=drift_weights.detach().max(),
            crsb_anchor_count=ref.new_tensor(
                float(len(self.sgr_semantic_anchor_prompts))))
        return loss, stats

    def _get_rvrd_cache(self) -> Optional[dict[str, Any]]:
        if self.sgr_rvrd_cache_file is None:
            return None
        cache = self._sgr_rvrd_cache
        if cache is not None:
            return cache
        payload = torch.load(str(self.sgr_rvrd_cache_file), map_location='cpu')
        expected = int(self.sgr_rvrd_expected_images)
        processed = int(payload.get('processed_images', 0))
        if expected > 0 and processed != expected:
            raise AssertionError(
                f'RVRD cache processed_images mismatch: {processed} != '
                f'{expected}')
        configured_teacher = self.sgr_rvrd_teacher
        cached_teacher = str(
            payload.get('rvrd_teacher', payload.get('remote_teacher', '')))
        if configured_teacher and cached_teacher:
            if configured_teacher.lower() != cached_teacher.lower():
                raise AssertionError(
                    f'RVRD cache teacher mismatch: {cached_teacher} != '
                    f'{configured_teacher}')
        feature_dim = int(payload.get('feature_dim', 0))
        raw_by_image = payload.get('by_image', {})
        by_image = {}
        for image_id, row in raw_by_image.items():
            boxes = torch.as_tensor(row['boxes'], dtype=torch.float32)
            tight = torch.as_tensor(row['tight_features'])
            context = torch.as_tensor(row['context_features'])
            if feature_dim <= 0 and tight.dim() == 2:
                feature_dim = int(tight.size(-1))
            if tight.numel() == 0:
                tight = torch.zeros((0, feature_dim), dtype=torch.float16)
            if context.numel() == 0:
                context = torch.zeros((0, feature_dim), dtype=torch.float16)
            by_image[int(image_id)] = dict(
                boxes=boxes,
                tight_features=tight,
                context_features=context,
                tight_context_cosine=torch.as_tensor(
                    row['tight_context_cosine'], dtype=torch.float32),
                generic_scores=torch.as_tensor(
                    row.get('generic_scores',
                            torch.zeros(len(boxes))),
                    dtype=torch.float32),
                query_indices=torch.as_tensor(
                    row.get('query_indices', torch.arange(len(boxes))),
                    dtype=torch.long))
        cache = dict(payload)
        cache['feature_dim'] = int(feature_dim)
        cache['by_image'] = by_image
        self._sgr_rvrd_cache = cache
        return cache

    def _match_rvrd_cache(self, sgr_matches: dict[str, Any],
                          batch_img_metas: List[dict],
                          ref: Tensor) -> dict[str, Tensor]:
        n = int(sgr_matches['student_feats'].size(0))
        device = ref.device
        cache = self._get_rvrd_cache()
        feature_dim = int(cache.get('feature_dim', 0)) if cache else 0
        out = dict(
            valid=torch.zeros(n, dtype=torch.bool, device=device),
            tight_features=ref.new_zeros((n, feature_dim)),
            context_features=ref.new_zeros((n, feature_dim)),
            rho=ref.new_zeros((n, )),
            boxes=ref.new_zeros((n, 4)),
            generic_scores=ref.new_zeros((n, )),
            query_indices=torch.full(
                (n, ), -1, dtype=torch.long, device=device),
            image_ids=torch.full((n, ), -1, dtype=torch.long, device=device),
            match_iou=ref.new_zeros((n, )),
            cache_count=ref.new_tensor(0.0),
            image_count=ref.new_tensor(float(len(batch_img_metas))))
        if n == 0 or cache is None:
            return out

        batch_idx = sgr_matches['match_batch_idx']
        teacher_boxes = sgr_matches['match_teacher_boxes']
        cache_count_total = 0
        for b, img_meta in enumerate(batch_img_metas):
            row_idx = torch.nonzero(
                batch_idx == int(b), as_tuple=False).flatten()
            if row_idx.numel() == 0:
                continue
            image_id = int(img_meta.get('img_id', img_meta.get('image_id')))
            cache_row = cache['by_image'].get(image_id, None)
            if cache_row is None:
                continue
            cache_boxes = cache_row['boxes'].to(device=device,
                                                dtype=ref.dtype)
            if cache_boxes.numel() == 0:
                continue
            cache_boxes = _boxes_original_to_current(cache_boxes, img_meta)
            cache_count_total += int(cache_boxes.size(0))
            ious = _bbox_overlaps_xyxy(teacher_boxes[row_idx].to(ref.dtype),
                                       cache_boxes)
            if ious.numel() == 0:
                continue
            best_iou, best_col = ious.max(dim=1)
            keep = best_iou >= self.sgr_rvrd_cache_match_iou
            if not bool(keep.any()):
                continue
            rows = row_idx[keep]
            cols = best_col[keep].detach().cpu()
            out['valid'][rows] = True
            out['image_ids'][rows] = int(image_id)
            out['match_iou'][rows] = best_iou[keep].detach()
            out['boxes'][rows] = cache_boxes[best_col[keep]].detach()
            out['generic_scores'][rows] = cache_row['generic_scores'][
                cols].to(device=device, dtype=ref.dtype)
            out['query_indices'][rows] = cache_row['query_indices'][
                cols].to(device=device)
            out['tight_features'][rows] = cache_row['tight_features'][
                cols].to(device=device, dtype=ref.dtype)
            out['context_features'][rows] = cache_row['context_features'][
                cols].to(device=device, dtype=ref.dtype)
            out['rho'][rows] = cache_row['tight_context_cosine'][
                cols].to(device=device, dtype=ref.dtype).clamp(0.0, 1.0)
        out['cache_count'] = ref.new_tensor(float(cache_count_total))
        return out

    def _deduplicate_rvrd_indices(self, rvrd_cache: dict[str, Tensor],
                                  ref: Tensor) -> Tensor:
        valid_idx = torch.nonzero(
            rvrd_cache['valid'], as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            return valid_idx
        device = ref.device
        image_ids = rvrd_cache['image_ids']
        boxes = rvrd_cache['boxes']
        scores = rvrd_cache['generic_scores']
        kept: list[int] = []
        unique_images = torch.unique(image_ids[valid_idx]).detach().cpu()
        for image_id in unique_images.tolist():
            rows = valid_idx[image_ids[valid_idx] == int(image_id)]
            if rows.numel() == 0:
                continue
            order = torch.argsort(scores[rows], descending=True)
            rows = rows[order]
            kept_image: list[int] = []
            for row_tensor in rows:
                row = int(row_tensor.detach().item())
                if kept_image:
                    prev = torch.as_tensor(
                        kept_image, dtype=torch.long, device=device)
                    iou = _bbox_overlaps_xyxy(boxes[row:row + 1],
                                              boxes[prev]).max()
                    if bool(iou.detach() > self.sgr_rvrd_duplicate_iou):
                        continue
                kept_image.append(row)
            kept.extend(kept_image)
        if not kept:
            return valid_idx.new_zeros((0, ))
        return torch.as_tensor(kept, dtype=torch.long, device=device)

    def _rvrd_empty_stats(self, ref: Tensor, rvrd_cache: dict[str, Tensor],
                          dedup_count: int) -> dict[str, Tensor]:
        n = int(rvrd_cache['valid'].numel()) if rvrd_cache else 0
        valid = rvrd_cache['valid'].detach().float() if rvrd_cache else \
            ref.new_zeros((0, ))
        image_count = (
            rvrd_cache.get('image_count', ref.new_tensor(1.0)).to(
                device=ref.device, dtype=ref.dtype) if rvrd_cache else
            ref.new_tensor(1.0))
        return dict(
            rvrd_valid_queries_per_img=valid.sum() /
            image_count.clamp_min(1.0),
            rvrd_deduplicated_queries_per_img=ref.new_tensor(
                float(dedup_count)) / image_count.clamp_min(1.0),
            rvrd_relation_pairs_per_batch=ref.new_tensor(0.0),
            rvrd_cache_hit_rate=valid.mean() if n > 0 else ref.new_tensor(0.0),
            rvrd_valid_query_count=valid.sum()
            if n > 0 else ref.new_tensor(0.0),
            rvrd_dedup_query_count=ref.new_tensor(float(dedup_count)),
            rvrd_teacher_relation_entropy=ref.new_tensor(0.0),
            rvrd_student_relation_entropy=ref.new_tensor(0.0),
            rvrd_relation_kl=ref.new_tensor(0.0),
            rvrd_tight_context_cosine=ref.new_tensor(0.0),
            rvrd_weight_mean=ref.new_tensor(0.0),
            rvrd_weight_max=ref.new_tensor(0.0),
            rvrd_zero_streak=ref.new_tensor(float(self._sgr_rvrd_zero_streak)))

    def _record_rvrd_effective_count(self, count: int) -> None:
        if not self.training or self.sgr_lambda_rvrd == 0.0:
            return
        if int(count) <= 0:
            self._sgr_rvrd_zero_streak += 1
        else:
            self._sgr_rvrd_zero_streak = 0
        if (self.sgr_rvrd_max_zero_streak > 0
                and self._sgr_rvrd_zero_streak >=
                self.sgr_rvrd_max_zero_streak):
            raise RuntimeError(
                'RVRD valid-query stream is empty for '
                f'{self._sgr_rvrd_zero_streak} consecutive batches.')

    def _loss_rvrd(self, student_feats: Tensor,
                   rvrd_cache: dict[str, Tensor],
                   ref: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        zero = ref.sum() * 0.0
        if student_feats.numel() == 0 or not rvrd_cache:
            self._record_rvrd_effective_count(0)
            return zero, self._rvrd_empty_stats(ref, rvrd_cache, 0)
        keep = self._deduplicate_rvrd_indices(rvrd_cache, ref)
        effective = int(keep.numel())
        if effective < self.sgr_rvrd_min_queries:
            self._record_rvrd_effective_count(0)
            return zero, self._rvrd_empty_stats(ref, rvrd_cache, effective)
        self._record_rvrd_effective_count(effective)

        eps = self.sgr_scr_eps
        tau = max(self.sgr_rvrd_temperature, eps)
        z_s = F.normalize(student_feats[keep].float(), dim=-1)
        tight = F.normalize(
            rvrd_cache['tight_features'][keep].float(), dim=-1)
        context = F.normalize(
            rvrd_cache['context_features'][keep].float(), dim=-1)
        z_t = F.normalize(tight + context, dim=-1).detach()
        weights = rvrd_cache['rho'][keep].float().clamp(0.0, 1.0).detach()
        weight_norm = weights / weights.mean().clamp_min(eps)

        teacher_rel = z_t @ z_t.T
        student_rel = z_s @ z_s.T
        diag = torch.eye(
            effective, dtype=torch.bool, device=student_rel.device)
        teacher_logits = (teacher_rel / tau).masked_fill(diag, -torch.inf)
        student_logits = (student_rel / tau).masked_fill(diag, -torch.inf)
        p_t = F.softmax(teacher_logits, dim=-1).detach().masked_fill(
            diag, 0.0)
        log_p_s = F.log_softmax(student_logits, dim=-1).masked_fill(
            diag, 0.0)
        p_s = F.softmax(student_logits, dim=-1).detach().masked_fill(
            diag, 0.0)
        kl = (p_t * ((p_t + eps).log() - log_p_s)).sum(dim=-1)
        norm = math.log(max(effective - 1, 2))
        unweighted = kl.mean() / norm
        loss = (kl * weight_norm).mean() / norm

        entropy_t = -(p_t * (p_t + eps).log()).sum(dim=-1)
        entropy_s = -(p_s * (p_s + eps).log()).sum(dim=-1)
        image_count = rvrd_cache['image_count'].to(
            device=ref.device, dtype=ref.dtype)
        valid_float = rvrd_cache['valid'].detach().float()
        stats = dict(
            rvrd_valid_queries_per_img=valid_float.sum() /
            image_count.clamp_min(1.0),
            rvrd_deduplicated_queries_per_img=ref.new_tensor(
                float(effective)) / image_count.clamp_min(1.0),
            rvrd_relation_pairs_per_batch=ref.new_tensor(
                float(effective * (effective - 1))),
            rvrd_cache_hit_rate=valid_float.mean(),
            rvrd_valid_query_count=valid_float.sum(),
            rvrd_dedup_query_count=ref.new_tensor(float(effective)),
            rvrd_teacher_relation_entropy=entropy_t.detach().mean(),
            rvrd_student_relation_entropy=entropy_s.detach().mean(),
            rvrd_relation_kl=kl.detach().mean(),
            rvrd_tight_context_cosine=weights.detach().mean(),
            rvrd_weight_mean=weights.detach().mean(),
            rvrd_weight_max=weights.detach().max(),
            rvrd_zero_streak=ref.new_tensor(float(self._sgr_rvrd_zero_streak)),
            rvrd_unweighted_normalized=unweighted.detach())
        return loss, stats

    def _rvrd_grad_audit_stats(self, losses: dict[str, Tensor],
                               matching_feats: Tensor, weighted_scr: Tensor,
                               weighted_rvrd: Tensor,
                               ref: Tensor) -> dict[str, Tensor]:
        interval = int(self.sgr_rvrd_grad_audit_interval)
        if (not self.training or interval <= 0
                or self._sgr_iter % interval != 0
                or not matching_feats.requires_grad):
            return {}
        aux_names = {
            'loss_gfr', 'loss_grr', 'loss_scr', 'loss_crsb', 'loss_rvrd'
        }
        main_terms = [
            value for key, value in losses.items()
            if torch.is_tensor(value) and 'loss' in key
            and key not in aux_names
        ]
        if not main_terms:
            return {}
        main_loss = sum(main_terms)

        def grad_or_none(loss_value: Tensor) -> Optional[Tensor]:
            if not torch.is_tensor(loss_value) or not loss_value.requires_grad:
                return None
            grad = torch.autograd.grad(
                loss_value,
                matching_feats,
                retain_graph=True,
                allow_unused=True)[0]
            return grad

        def norm_or_zero(grad: Optional[Tensor]) -> Tensor:
            if grad is None:
                return ref.new_tensor(0.0)
            return grad.detach().float().norm()

        def cosine_or_zero(a: Optional[Tensor],
                           b: Optional[Tensor]) -> Tensor:
            if a is None or b is None:
                return ref.new_tensor(0.0)
            a_flat = a.detach().float().reshape(-1)
            b_flat = b.detach().float().reshape(-1)
            denom = a_flat.norm() * b_flat.norm()
            if bool(denom <= self.sgr_scr_eps):
                return ref.new_tensor(0.0)
            return (a_flat * b_flat).sum() / denom.clamp_min(
                self.sgr_scr_eps)

        grad_main = grad_or_none(main_loss)
        grad_scr = grad_or_none(weighted_scr)
        grad_rvrd = grad_or_none(weighted_rvrd)
        main_norm = norm_or_zero(grad_main)
        scr_norm = norm_or_zero(grad_scr)
        rvrd_norm = norm_or_zero(grad_rvrd)
        return dict(
            rvrd_classifier_local_grad_norm_main=main_norm,
            rvrd_classifier_local_grad_norm_scr=scr_norm,
            rvrd_classifier_local_grad_norm_rvrd=rvrd_norm,
            rvrd_classifier_local_grad_ratio=rvrd_norm /
            main_norm.clamp_min(self.sgr_scr_eps),
            rvrd_to_scr_grad_ratio=rvrd_norm /
            scr_norm.clamp_min(self.sgr_scr_eps),
            rvrd_scr_grad_cosine=cosine_or_zero(grad_rvrd, grad_scr))

    def _prepare_text_for_loss(self, batch_inputs: Tensor,
                               batch_data_samples: SampleList) -> dict:
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]
        enhanced_text_prompts = []
        for data_samples in batch_data_samples:
            enhanced_text_prompts.append(
                data_samples.caption_prompt
                if 'caption_prompt' in data_samples else None)

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        # Paper-compatible DVC cap with a corrected per-instance label map.
        # Every positive class is retained; remaining slots are sampled from
        # negatives.  MMEngine seeds Python's RNG on every distributed rank;
        # exact mid-epoch resume equivalence is not claimed for this sampler.
        if self.sgr_dvc_max_classes > 0:
            sampled_prompts = []
            sampled_labels = []
            for text_prompt, gt_label in zip(text_prompts, gt_labels):
                prompt_tuple = tuple(text_prompt)
                if len(prompt_tuple) <= self.sgr_dvc_max_classes:
                    sampled_prompts.append(prompt_tuple)
                    sampled_labels.append(gt_label)
                    continue
                positives = sorted(set(int(x) for x in
                                       gt_label.detach().cpu().tolist()))
                if len(positives) > self.sgr_dvc_max_classes:
                    raise ValueError(
                        'positive classes exceed the configured DVC cap: '
                        f'{len(positives)} > {self.sgr_dvc_max_classes}')
                positive_set = set(positives)
                negatives = [i for i in range(len(prompt_tuple))
                             if i not in positive_set]
                need = self.sgr_dvc_max_classes - len(positives)
                chosen = positives + random.sample(negatives, need)
                remap = {old: new for new, old in enumerate(chosen)}
                sampled_prompts.append(tuple(prompt_tuple[i] for i in chosen))
                sampled_labels.append(torch.as_tensor(
                    [remap[int(x)] for x in gt_label.detach().cpu().tolist()],
                    dtype=gt_label.dtype, device=gt_label.device))
            text_prompts = sampled_prompts
            gt_labels = sampled_labels

        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            same_text_prompt = len(set(text_prompts)) == 1
            same_enhanced_prompt = all(
                enhanced_text_prompt == enhanced_text_prompts[0]
                for enhanced_text_prompt in enhanced_text_prompts)
            if same_text_prompt and same_enhanced_prompt:
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True, enhanced_text_prompts[0])
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label, enhanced_text_prompt in zip(
                        text_prompts, gt_labels, enhanced_text_prompts):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True, enhanced_text_prompt)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(len(positive_map), 1)
        return text_dict

    def _budget_class_maps(self,
                           batch_data_samples: SampleList) -> list[dict]:
        """Rebuild the complete prompt-to-token maps used at inference.

        Grounding DINO's ordinary training target stores maps only for GT
        labels present in an image.  Budget priority, however, is the maximum
        over every prompted class, so this helper reconstructs the complete
        class map without another language-model forward.
        """
        maps = []
        for data_sample in batch_data_samples:
            custom_entities = bool(data_sample.custom_entities) \
                if 'custom_entities' in data_sample else False
            caption_prompt = data_sample.caption_prompt \
                if 'caption_prompt' in data_sample else None
            tokens_positive = data_sample.get('tokens_positive', None)
            token_map, normalized_prompt, _, _ = \
                self.get_tokens_positive_and_prompts(
                    data_sample.text, custom_entities, caption_prompt,
                    tokens_positive)
            if isinstance(normalized_prompt, list):
                raise NotImplementedError(
                    'chunked prompts are unsupported by joint budget loss')
            maps.append(token_map)
        if maps and any(len(row) != len(maps[0]) for row in maps):
            raise AssertionError(
                'joint budget loss requires a shared class prompt per batch')
        return maps

    def _loss_budget_coverage(
            self, matching_cls: Tensor, matching_boxes: Tensor,
            batch_gt_instances: InstanceList, batch_img_metas: List[dict],
            batch_data_samples: SampleList) -> tuple[Tensor, dict[str, Any]]:
        """Unique-GT soft Top-50 coverage on the live 900-query scores.

        IoU support membership is detached and used only to name which query
        can represent each annotated base object.  The differentiable path is
        exclusively through the existing class scores.  Consequently this
        loss can reshape the original detector representation but cannot
        directly move a box through the hard IoU threshold.
        """
        token_maps = self._budget_class_maps(batch_data_samples)
        class_scores = convert_grounding_to_cls_scores(
            logits=matching_cls.sigmoid(), positive_maps=token_maps)
        objectness = class_scores.max(dim=-1).values
        priorities = torch.logit(objectness.clamp(1e-5, 1.0 - 1e-5))
        image_losses = []
        represented_gt = 0
        total_gt = 0
        support_queries = 0
        occupied = []
        mean_coverage = []
        for image_idx, (priority, box_pred, gt_instances, img_meta) in \
                enumerate(zip(priorities, matching_boxes,
                              batch_gt_instances, batch_img_metas)):
            del image_idx
            gt_bboxes = gt_instances.bboxes
            if hasattr(gt_bboxes, 'tensor'):
                gt_bboxes = gt_bboxes.tensor
            gt_bboxes = gt_bboxes.to(
                device=box_pred.device, dtype=box_pred.dtype)
            total_gt += int(gt_bboxes.shape[0])
            if gt_bboxes.numel() == 0:
                continue
            with torch.no_grad():
                current_boxes = _boxes_current_xyxy(
                    box_pred.detach(), img_meta)
                overlaps = _bbox_overlaps_xyxy(current_boxes, gt_bboxes)
                support_matrix = overlaps >= self.sgr_budget_iou_thr
            if not bool(support_matrix.any()):
                continue
            weights = _soft_budget(
                priority, self.sgr_budget, self.sgr_budget_temperature)
            occupied.append(weights.sum().detach())
            per_gt = []
            for gt_idx in range(int(gt_bboxes.shape[0])):
                mask = support_matrix[:, gt_idx]
                if bool(mask.any()):
                    per_gt.append(weights[mask].max())
            if not per_gt:
                continue
            coverage = torch.stack(per_gt)
            image_losses.append(-torch.log(
                1e-6 + (1.0 - 1e-6) * coverage).mean())
            represented_gt += len(per_gt)
            support_queries += int(support_matrix.any(dim=1).sum().detach())
            mean_coverage.append(coverage.mean().detach())
        # ``matching_cls`` contains -inf at masked text-token positions, so
        # multiplying its sum by zero would make an otherwise valid empty-
        # support batch NaN.  ``priorities`` is finite by construction and
        # retains a zero-gradient connection to the live classification path.
        zero = priorities.sum() * 0.0
        loss = torch.stack(image_losses).mean() if image_losses else zero
        stats = dict(
            images_with_support=len(image_losses),
            represented_gt=int(represented_gt),
            total_gt=int(total_gt),
            support_queries=int(support_queries),
            mean_soft_occupancy=float(torch.stack(occupied).mean().cpu())
            if occupied else 0.0,
            mean_gt_coverage=float(torch.stack(mean_coverage).mean().cpu())
            if mean_coverage else 0.0)
        return loss, stats

    @staticmethod
    def _geo_expand_boxes(boxes: Tensor, scale: float,
                          img_h: int, img_w: int) -> Tensor:
        centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        sizes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1.0) * scale
        out = torch.cat([centers - sizes * 0.5,
                         centers + sizes * 0.5], dim=-1)
        out[:, 0::2].clamp_(0, float(img_w))
        out[:, 1::2].clamp_(0, float(img_h))
        return out

    def predict_geo_quality(self, visual_features, boxes_per_image,
                            batch_img_metas: List[dict],
                            chunk_size: int = 256) -> list[Tensor]:
        """Predict q for current-image xyxy boxes without changing boxes."""
        if self.geo_head is None:
            raise RuntimeError('GEO head is not enabled')
        feat8, feat16 = self._geo_input_features(visual_features)
        input_h = max(int(m['batch_input_shape'][0])
                      if 'batch_input_shape' in m else int(feat8.shape[-2] * 8)
                      for m in batch_img_metas)
        input_w = max(int(m['batch_input_shape'][1])
                      if 'batch_input_shape' in m else int(feat8.shape[-1] * 8)
                      for m in batch_img_metas)
        outputs = []
        for image_idx, (boxes, meta) in enumerate(zip(
                boxes_per_image, batch_img_metas)):
            img_h, img_w = meta['img_shape'][:2]
            expanded = self._geo_expand_boxes(
                boxes, self.geo_roi_scale, int(img_h), int(img_w))
            chunks = []
            for start in range(0, int(expanded.shape[0]), int(chunk_size)):
                part = expanded[start:start + int(chunk_size)]
                rois = torch.cat([
                    part.new_full((part.shape[0], 1), float(image_idx)), part
                ], dim=1)
                _, q = self.geo_head(feat8, feat16, rois,
                                     (input_h, input_w))
                chunks.append(q)
            outputs.append(torch.cat(chunks) if chunks else
                           boxes.new_zeros((0, )))
        return outputs

    @staticmethod
    def _geo_iou_aligned(boxes: Tensor, gt: Tensor) -> Tensor:
        lt = torch.maximum(boxes[:, :2], gt[:, :2])
        rb = torch.minimum(boxes[:, 2:], gt[:, 2:])
        wh = (rb - lt).clamp_min(0)
        inter = wh[:, 0] * wh[:, 1]
        a = ((boxes[:, 2] - boxes[:, 0]).clamp_min(0) *
             (boxes[:, 3] - boxes[:, 1]).clamp_min(0))
        b = ((gt[:, 2] - gt[:, 0]).clamp_min(0) *
             (gt[:, 3] - gt[:, 1]).clamp_min(0))
        return inter / (a + b - inter).clamp_min(1e-6)

    @staticmethod
    def _geo_giou_loss(boxes: Tensor, gt: Tensor) -> Tensor:
        lt = torch.maximum(boxes[:, :2], gt[:, :2])
        rb = torch.minimum(boxes[:, 2:], gt[:, 2:])
        wh = (rb - lt).clamp_min(0)
        inter = wh[:, 0] * wh[:, 1]
        a = ((boxes[:, 2] - boxes[:, 0]).clamp_min(0) *
             (boxes[:, 3] - boxes[:, 1]).clamp_min(0))
        b = ((gt[:, 2] - gt[:, 0]).clamp_min(0) *
             (gt[:, 3] - gt[:, 1]).clamp_min(0))
        union = (a + b - inter).clamp_min(1e-6)
        iou = inter / union
        enc_lt = torch.minimum(boxes[:, :2], gt[:, :2])
        enc_rb = torch.maximum(boxes[:, 2:], gt[:, 2:])
        enc_wh = (enc_rb - enc_lt).clamp_min(0)
        enc = (enc_wh[:, 0] * enc_wh[:, 1]).clamp_min(1e-6)
        return 1.0 - (iou - (enc - union) / enc)

    @staticmethod
    def _geo_encode(boxes: Tensor, gt: Tensor) -> Tensor:
        bc = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        bs = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-4)
        gc = (gt[:, :2] + gt[:, 2:]) * 0.5
        gs = (gt[:, 2:] - gt[:, :2]).clamp_min(1e-4)
        return torch.cat([(gc - bc) / bs, torch.log(gs / bs)], dim=-1)

    @staticmethod
    def _geo_decode(boxes: Tensor, delta: Tensor) -> Tensor:
        bc = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        bs = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-4)
        pc = bc + delta[:, :2] * bs
        ps = bs * delta[:, 2:].clamp(-4.0, 4.0).exp()
        return torch.cat([pc - ps * 0.5, pc + ps * 0.5], dim=-1)

    def _geo_select_gt(self, gt: Tensor, generator: torch.Generator) -> Tensor:
        count = int(gt.shape[0])
        if count <= self.geo_max_gt:
            return torch.arange(count, dtype=torch.long)
        side = (((gt[:, 2] - gt[:, 0]).clamp_min(0) *
                 (gt[:, 3] - gt[:, 1]).clamp_min(0)).sqrt()).cpu()
        groups = [torch.where(side < 32)[0],
                  torch.where((side >= 32) & (side < 96))[0],
                  torch.where(side >= 96)[0]]
        chosen = []
        while len(chosen) < self.geo_max_gt and any(len(g) for g in groups):
            for idx, group in enumerate(groups):
                if len(chosen) >= self.geo_max_gt or len(group) == 0:
                    continue
                pos = int(torch.randint(len(group), (1,), generator=generator))
                chosen.append(int(group[pos]))
                groups[idx] = torch.cat([group[:pos], group[pos + 1:]])
        return torch.tensor(chosen, dtype=torch.long)

    def _geo_perturb_one(self, gt_box: Tensor, all_gt: Tensor,
                         interval: tuple[float, float],
                         generator: torch.Generator,
                         img_h: int, img_w: int) -> Optional[Tensor]:
        low, high = interval
        if low >= 0.9:
            return gt_box.clone()
        box_cpu = gt_box.detach().float().cpu()
        all_cpu = all_gt.detach().float().cpu()
        center = (box_cpu[:2] + box_cpu[2:]) * 0.5
        size = (box_cpu[2:] - box_cpu[:2]).clamp_min(2.0)
        for _ in range(32):
            offset = (torch.rand(2, generator=generator) * 0.8 - 0.4) * size
            scale = 0.6 + torch.rand(2, generator=generator)
            c = center + offset
            s = size * scale
            candidate = torch.cat([c - s * 0.5, c + s * 0.5])
            candidate[0::2].clamp_(0, float(img_w))
            candidate[1::2].clamp_(0, float(img_h))
            if bool(((candidate[2:] - candidate[:2]) < 2.0).any()):
                continue
            overlaps = _bbox_overlaps_xyxy(candidate[None], all_cpu)[0]
            best_iou, best_idx = overlaps.max(dim=0)
            if int(best_idx) != int(torch.argmin(
                    ((all_cpu - box_cpu) ** 2).sum(dim=1))):
                continue
            value = float(best_iou)
            if value >= low and (value < high or high >= 1.0):
                return candidate.to(device=gt_box.device, dtype=gt_box.dtype)
        return None

    def _loss_geo_v1(self, visual_features, matching_boxes: Tensor,
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict]) -> tuple[Tensor, dict]:
        if self.geo_head is None:
            return matching_boxes.sum() * 0.0, {}
        feat8, feat16 = self._geo_input_features(visual_features)
        input_h, input_w = int(feat8.shape[-2] * 8), int(feat8.shape[-1] * 8)
        image_losses, q_values, targets = [], [], []
        synth_count = real_count = 0
        intervals = ((0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0))
        for image_idx, (box_pred, gt_instances, meta) in enumerate(zip(
                matching_boxes, batch_gt_instances, batch_img_metas)):
            gt = gt_instances.bboxes
            if hasattr(gt, 'tensor'):
                gt = gt.tensor
            gt = gt.to(device=box_pred.device, dtype=box_pred.dtype)
            if gt.numel() == 0:
                continue
            gen = torch.Generator(device='cpu')
            gen.manual_seed(self.geo_seed + self._sgr_iter * 10007 +
                            image_idx * 97)
            selected = self._geo_select_gt(gt, gen).to(gt.device)
            sample_boxes, sample_gt, object_ids, source_ids = [], [], [], []
            img_h, img_w = meta['img_shape'][:2]
            for local_id, gt_idx_tensor in enumerate(selected):
                gt_idx = int(gt_idx_tensor)
                for interval in intervals:
                    candidate = self._geo_perturb_one(
                        gt[gt_idx], gt, interval, gen, int(img_h), int(img_w))
                    if candidate is not None:
                        sample_boxes.append(candidate)
                        sample_gt.append(gt[gt_idx])
                        object_ids.append(local_id)
                        source_ids.append(0)
                        synth_count += 1
            current = _boxes_current_xyxy(box_pred.detach(), meta)
            overlaps = _bbox_overlaps_xyxy(current, gt)
            best_iou, best_gt = overlaps.max(dim=1)
            real_indices = []
            for low, high in intervals:
                mask = (best_iou >= low) & ((best_iou < high) if high < 1.0
                                            else (best_iou <= high))
                candidates = torch.where(mask)[0].cpu()
                if candidates.numel():
                    perm = torch.randperm(candidates.numel(), generator=gen)
                    real_indices.extend(candidates[perm[:16]].tolist())
            real_indices = real_indices[:self.geo_max_real]
            for query_idx in real_indices:
                sample_boxes.append(current[query_idx])
                sample_gt.append(gt[int(best_gt[query_idx])])
                object_ids.append(int(best_gt[query_idx]) + self.geo_max_gt)
                source_ids.append(1)
                real_count += 1
            if not sample_boxes:
                continue
            boxes = torch.stack(sample_boxes)
            target_gt = torch.stack(sample_gt)
            expanded = self._geo_expand_boxes(
                boxes, self.geo_roi_scale, int(img_h), int(img_w))
            rois = torch.cat([
                expanded.new_full((expanded.shape[0], 1), float(image_idx)),
                expanded], dim=1)
            delta, quality = self.geo_head(feat8, feat16, rois,
                                           (input_h, input_w))
            target_q = self._geo_iou_aligned(boxes, target_gt).detach()
            target_delta = self._geo_encode(boxes, target_gt).detach()
            refined = self._geo_decode(boxes, delta)
            row_loss = (self.geo_quality_weight * (quality - target_q).abs() +
                        self.geo_offset_weight * F.smooth_l1_loss(
                            delta, target_delta, beta=1.0,
                            reduction='none').mean(dim=1) +
                        self.geo_giou_weight * self._geo_giou_loss(
                            refined, target_gt))
            per_source = []
            object_tensor = torch.tensor(object_ids, device=boxes.device)
            source_tensor = torch.tensor(source_ids, device=boxes.device)
            for source in (0, 1):
                source_object_losses = []
                source_mask = source_tensor == source
                for object_id in object_tensor[source_mask].unique():
                    mask = source_mask & (object_tensor == object_id)
                    source_object_losses.append(row_loss[mask].mean())
                if source_object_losses:
                    per_source.append(torch.stack(source_object_losses).mean())
            if per_source:
                image_losses.append(torch.stack(per_source).mean())
            q_values.append(quality.detach())
            targets.append(target_q.detach())
        # Keep every GEO parameter in the DDP graph on empty-GT ranks.  The
        # value and gradients remain exactly zero, but bucket reduction stays
        # synchronized across ranks with different image contents.
        zero = matching_boxes.sum() * 0.0
        for parameter in self.geo_head.parameters():
            zero = zero + parameter.sum() * 0.0
        if self.geo_adapters is not None:
            for parameter in self.geo_adapters.parameters():
                zero = zero + parameter.sum() * 0.0
        loss = torch.stack(image_losses).mean() if image_losses else zero
        q_all = torch.cat(q_values) if q_values else matching_boxes.new_zeros(0)
        t_all = torch.cat(targets) if targets else matching_boxes.new_zeros(0)
        stats = dict(synthetic_samples=synth_count, real_samples=real_count,
                     quality_weight=self.geo_quality_weight,
                     offset_weight=self.geo_offset_weight,
                     giou_weight=self.geo_giou_weight,
                     mean_q=float(q_all.mean().cpu()) if q_all.numel() else 0.0,
                     mean_target=float(t_all.mean().cpu()) if t_all.numel() else 0.0)
        return loss, stats

    def _loss_novel_box_only(
            self, all_boxes: Tensor, num_dn: int,
            batch_data_samples: SampleList,
            batch_img_metas: List[dict]) -> tuple[Tensor, Tensor, dict]:
        """Anonymous one-to-one localization for ignored novel boxes.

        The selected decoder queries are returned as a protection mask so
        their ordinary all-class negative loss can be set to zero.  The
        Hungarian assignment itself is detached; localization gradients flow
        through every decoder layer at the selected query indices.
        """
        matching_layers = all_boxes[:, :, num_dn:, :]
        final_boxes = matching_layers[-1]
        protected = torch.zeros(
            final_boxes.shape[:2], dtype=torch.bool,
            device=final_boxes.device)
        layer_losses = []
        matched_boxes = 0
        images_with_novel = 0
        for image_idx, (pred, sample, meta) in enumerate(zip(
                final_boxes, batch_data_samples, batch_img_metas)):
            ignored = getattr(sample, 'ignored_instances', None)
            if ignored is None or len(ignored) == 0:
                continue
            gt_xyxy = ignored.bboxes
            if hasattr(gt_xyxy, 'tensor'):
                gt_xyxy = gt_xyxy.tensor
            gt_xyxy = gt_xyxy.to(device=pred.device, dtype=pred.dtype)
            if gt_xyxy.numel() == 0:
                continue
            img_h, img_w = meta['img_shape'][:2]
            factor = pred.new_tensor([img_w, img_h, img_w, img_h])
            gt_xyxy_norm = gt_xyxy / factor
            gt_cxcywh = bbox_xyxy_to_cxcywh(gt_xyxy_norm)
            pred_xyxy = bbox_cxcywh_to_xyxy(pred)
            with torch.no_grad():
                cost_l1 = torch.cdist(pred, gt_cxcywh, p=1)
                cost_giou = -bbox_overlaps(
                    pred_xyxy, gt_xyxy_norm, mode='giou', is_aligned=False)
                row, col = linear_sum_assignment(
                    (cost_l1 + 2.0 * cost_giou).float().cpu().numpy())
                query_idx = torch.as_tensor(
                    row, dtype=torch.long, device=pred.device)
                gt_idx = torch.as_tensor(
                    col, dtype=torch.long, device=pred.device)
            if query_idx.numel() == 0:
                continue
            protected[image_idx, query_idx] = True
            matched_boxes += int(query_idx.numel())
            images_with_novel += 1
            per_layer = []
            for layer in matching_layers:
                selected = layer[image_idx, query_idx]
                target = gt_cxcywh[gt_idx]
                loss_l1 = F.l1_loss(
                    selected, target, reduction='none').sum(dim=-1).mean()
                giou = bbox_overlaps(
                    bbox_cxcywh_to_xyxy(selected),
                    gt_xyxy_norm[gt_idx], mode='giou', is_aligned=True)
                loss_giou = (1.0 - giou).mean()
                per_layer.append(
                    self.novel_box_bbox_weight * loss_l1 +
                    self.novel_box_giou_weight * loss_giou)
            layer_losses.append(torch.stack(per_layer).mean())
        zero = final_boxes.sum() * 0.0
        loss = torch.stack(layer_losses).mean() if layer_losses else zero
        return loss, protected, {
            'images_with_novel': images_with_novel,
            'matched_boxes': matched_boxes,
            'protected_queries': int(protected.sum().detach().item())}

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        self._sgr_iter += 1
        text_dict = self._prepare_text_for_loss(batch_inputs,
                                                batch_data_samples)
        visual_features = self.extract_feat(batch_inputs)
        head_inputs = self.forward_transformer(visual_features, text_dict,
                                               batch_data_samples)
        if self.keep_enabled:
            self._keep_student_dense_cls = head_inputs.get(
                'keep_student_dense_cls', self._keep_student_dense_cls)
            self._keep_student_memory_mask = head_inputs.get(
                'keep_student_memory_mask', self._keep_student_memory_mask)
        all_cls, all_boxes = self.bbox_head(
            head_inputs['hidden_states'],
            head_inputs['references'],
            head_inputs['memory_text'],
            head_inputs['text_token_mask'])

        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        num_dn = SGRGroundingDINOHead._get_num_dn_queries(
            head_inputs['dn_meta'])
        matching_feats = head_inputs['hidden_states'][-1][:, num_dn:, :]
        matching_cls = all_cls[-1][:, num_dn:, :]
        matching_boxes = all_boxes[-1][:, num_dn:, :]
        sgr_matches = None
        protected_mask = None
        protected_weights = None
        if self._should_run_sgr_path():
            if not isinstance(self.bbox_head, SGRGroundingDINOHead):
                raise TypeError('SGR path requires SGRGroundingDINOHead.')
            teacher_out = self._forward_sgr_teacher(
                batch_inputs, copy.deepcopy(batch_data_samples))
            sgr_matches = self._build_sgr_matches(
                teacher_out, matching_cls, matching_boxes, matching_feats,
                batch_gt_instances, batch_img_metas)
            protected_mask = sgr_matches['protected_mask']
            if self.sgr_soft_ubp_enabled:
                protected_weights = sgr_matches['protected_weights']
            self._last_sgr_protected_mask = protected_mask.detach().cpu()
            self._last_sgr_stats = sgr_matches['stats']

        if self.novel_box_enabled:
            loss_novel_box, novel_mask, novel_stats = \
                self._loss_novel_box_only(
                    all_boxes, num_dn, batch_data_samples, batch_img_metas)
            protected_mask = (novel_mask if protected_mask is None else
                              (protected_mask | novel_mask))
            losses_novel_box = loss_novel_box
            self._last_novel_box_stats = novel_stats
        else:
            losses_novel_box = None
            self._last_novel_box_stats = {}

        self.bbox_head.text_masks = head_inputs['text_token_mask']
        losses = self.bbox_head.loss_by_feat(
            all_cls,
            all_boxes,
            head_inputs['enc_outputs_class'],
            head_inputs['enc_outputs_coord'],
            batch_gt_instances,
            batch_img_metas,
            head_inputs['dn_meta'],
            sgr_protected_mask=protected_mask,
            sgr_protected_weights=protected_weights)

        if losses_novel_box is not None:
            losses['loss_novel_box'] = losses_novel_box

        if self.sgr_lambda_budget != 0.0:
            loss_budget_raw, budget_stats = self._loss_budget_coverage(
                matching_cls, matching_boxes, batch_gt_instances,
                batch_img_metas, batch_data_samples)
            budget_warmup = self._sgr_warmup_factor(
                matching_cls, self.sgr_budget_warmup_iters)
            losses['loss_budget'] = (
                loss_budget_raw * self.sgr_lambda_budget * budget_warmup)
            self._last_budget_stats = {
                **budget_stats,
                'raw_loss': float(loss_budget_raw.detach().float().cpu()),
                'weighted_loss': float(
                    losses['loss_budget'].detach().float().cpu()),
                'warmup': float(budget_warmup.detach().float().cpu())}
        else:
            self._last_budget_stats = {}

        if self.geo_enabled:
            loss_geo_raw, geo_stats = self._loss_geo_v1(
                visual_features, matching_boxes, batch_gt_instances,
                batch_img_metas)
            geo_warmup = matching_boxes.new_tensor(min(
                1.0, self._sgr_iter / max(1, self.geo_warmup_iters)))
            losses['loss_geo'] = loss_geo_raw * self.geo_lambda * geo_warmup
            self._last_geo_stats = dict(
                **geo_stats,
                raw_loss=float(loss_geo_raw.detach().float().cpu()),
                weighted_loss=float(losses['loss_geo'].detach().float().cpu()),
                warmup=float(geo_warmup.cpu()))
        else:
            self._last_geo_stats = {}

        if self.keep_enabled:
            loss_keep_raw, keep_stats = self._loss_task_adapted_keep(
                batch_inputs, batch_data_samples)
            losses['loss_keep'] = loss_keep_raw * self.keep_lambda
            self._last_keep_stats = {
                **keep_stats,
                'raw_loss': float(loss_keep_raw.detach().float().cpu()),
                'weighted_loss': float(
                    losses['loss_keep'].detach().float().cpu())}
        else:
            self._last_keep_stats = {}

        if sgr_matches is not None:
            if self.sgr_log_layer_audit:
                self._last_sgr_stats['protected_layer_correspondence'] = (
                    self._audit_protected_layer_correspondence(
                        all_boxes, num_dn, sgr_matches, batch_img_metas))
            if getattr(self.bbox_head, 'sgr_audit_ubp_delta', False):
                self._last_sgr_stats['ubp_decoder_layers'] = copy.deepcopy(
                    self.bbox_head._sgr_ubp_audit_rows)
            warmup = self._sgr_warmup_factor(matching_feats)
            need_gfr = self.sgr_lambda_gfr != 0.0 or self.sgr_log_loss_keys
            need_grr = self.sgr_lambda_grr != 0.0 or self.sgr_log_loss_keys
            need_rank = self.sgr_lambda_rank != 0.0 or self.sgr_log_loss_keys
            need_scr = self.sgr_lambda_scr != 0.0 or self.sgr_log_scr_keys
            need_crsb = (
                self.sgr_lambda_crsb != 0.0 or self.sgr_log_crsb_keys)
            need_rvrd = (
                self.sgr_lambda_rvrd != 0.0 or self.sgr_log_rvrd_keys)
            loss_gfr_raw = (
                self._loss_gfr(sgr_matches['student_feats'],
                               sgr_matches['teacher_feats'],
                               sgr_matches['match_weights'], matching_feats)
                if need_gfr else matching_feats.sum() * 0.0)
            loss_grr_raw = (
                self._loss_grr(matching_feats, protected_mask,
                               sgr_matches['base_pos_mask'], matching_cls,
                               head_inputs['text_token_mask'])
                if need_grr else matching_feats.sum() * 0.0)
            loss_rank_raw = (
                self._loss_rank_retention(
                    matching_feats, protected_mask,
                    sgr_matches['protected_weights'],
                    sgr_matches['base_pos_mask'])
                if need_rank else matching_feats.sum() * 0.0)
            loss_scr_raw, scr_stats = (
                self._loss_scr(sgr_matches['student_feats'],
                               sgr_matches['teacher_feats'], matching_feats)
                if need_scr else (matching_feats.sum() * 0.0, {}))
            crsb_cache = (
                self._match_crsb_cache(sgr_matches, batch_img_metas,
                                       matching_feats)
                if need_crsb else {})
            loss_crsb_raw, crsb_stats = (
                self._loss_crsb(sgr_matches['student_feats'],
                                sgr_matches['teacher_feats'], crsb_cache,
                                matching_feats)
                if need_crsb else (matching_feats.sum() * 0.0, {}))
            rvrd_cache = (
                self._match_rvrd_cache(sgr_matches, batch_img_metas,
                                       matching_feats)
                if need_rvrd else {})
            loss_rvrd_raw, rvrd_stats = (
                self._loss_rvrd(sgr_matches['student_feats'], rvrd_cache,
                                matching_feats)
                if need_rvrd else (matching_feats.sum() * 0.0, {}))
            weighted_gfr = loss_gfr_raw * self.sgr_lambda_gfr * warmup
            weighted_grr = loss_grr_raw * self.sgr_lambda_grr * warmup
            weighted_rank = loss_rank_raw * self.sgr_lambda_rank * warmup
            weighted_scr = loss_scr_raw * self.sgr_lambda_scr * warmup
            weighted_crsb = loss_crsb_raw * self.sgr_lambda_crsb * warmup
            rvrd_warmup = self._sgr_warmup_factor(
                matching_feats, self.sgr_rvrd_warmup_iters)
            weighted_rvrd = (
                loss_rvrd_raw * self.sgr_lambda_rvrd * rvrd_warmup)
            if self.sgr_lambda_gfr != 0.0:
                losses['loss_gfr'] = weighted_gfr
            if self.sgr_lambda_grr != 0.0:
                losses['loss_grr'] = weighted_grr
            if self.sgr_lambda_rank != 0.0:
                losses['loss_rank_retention'] = weighted_rank
            if self.sgr_lambda_scr != 0.0:
                losses['loss_scr'] = weighted_scr
            if self.sgr_lambda_crsb != 0.0:
                losses['loss_crsb'] = weighted_crsb
            if self.sgr_lambda_rvrd != 0.0:
                losses['loss_rvrd'] = weighted_rvrd
            if rvrd_stats:
                for key, value in rvrd_stats.items():
                    if torch.is_tensor(value):
                        self._last_sgr_stats[key] = float(
                            value.detach().float().cpu())
                    else:
                        self._last_sgr_stats[key] = value
            if self.sgr_debug_stats and self.sgr_log_loss_keys:
                stats = sgr_matches['stats']
                candidate = float(sum(stats.get('candidate_counts', [])))
                selected = float(sum(stats.get('selected_counts', [])))
                losses['sgr_num_protected'] = matching_feats.new_tensor(
                    float(protected_mask.sum().detach()))
                losses['sgr_num_candidate'] = matching_feats.new_tensor(
                    candidate)
                losses['sgr_num_selected'] = matching_feats.new_tensor(
                    selected)
                losses['sgr_cap_hit'] = matching_feats.new_tensor(
                    float(sum(x >= self.sgr_max_protected
                              for x in stats.get('candidate_counts', []))))
                losses['sgr_protected_base_pos_overlap'] = (
                    matching_feats.new_tensor(
                        float(stats.get('protected_base_pos_overlap', 0))))
                losses['unweighted_gfr'] = loss_gfr_raw.detach()
                losses['weighted_gfr'] = weighted_gfr.detach()
                losses['unweighted_grr'] = loss_grr_raw.detach()
                losses['weighted_grr'] = weighted_grr.detach()
                losses['unweighted_rank_retention'] = loss_rank_raw.detach()
                losses['weighted_rank_retention'] = weighted_rank.detach()
                losses['unweighted_crsb'] = loss_crsb_raw.detach()
                losses['weighted_crsb'] = weighted_crsb.detach()
                losses['unweighted_rvrd'] = loss_rvrd_raw.detach()
                losses['weighted_rvrd'] = weighted_rvrd.detach()
                losses['sgr_warmup'] = warmup.detach()
            if self.sgr_debug_stats and self.sgr_log_scr_keys:
                losses['unweighted_scr'] = loss_scr_raw.detach()
                losses['weighted_scr'] = weighted_scr.detach()
                for key, value in scr_stats.items():
                    losses[key] = value.detach()
                grad_scale = (
                    matching_feats.new_tensor(self.sgr_lambda_scr) * warmup)
                if 'scr_feature_grad_norm_raw' in scr_stats:
                    losses['scr_feature_grad_norm_effective'] = (
                        scr_stats['scr_feature_grad_norm_raw'].detach()
                        * grad_scale.detach())
                    losses['scr_feature_grad_rms_effective'] = (
                        scr_stats['scr_feature_grad_rms_raw'].detach()
                        * grad_scale.detach())
            if self.sgr_debug_stats and self.sgr_log_crsb_keys:
                losses['unweighted_crsb'] = loss_crsb_raw.detach()
                losses['weighted_crsb'] = weighted_crsb.detach()
                for key, value in crsb_stats.items():
                    losses[key] = value.detach()
            if self.sgr_debug_stats and self.sgr_log_rvrd_keys:
                losses['unweighted_rvrd'] = loss_rvrd_raw.detach()
                losses['weighted_rvrd'] = weighted_rvrd.detach()
                losses['rvrd_warmup'] = rvrd_warmup.detach()
                for key, value in rvrd_stats.items():
                    losses[key] = value.detach()
                for key, value in self._rvrd_grad_audit_stats(
                        losses, head_inputs['hidden_states'][-1],
                        weighted_scr, weighted_rvrd, matching_feats).items():
                    losses[key] = value.detach()
            if (self.sgr_scr_direction_audit_tensors and
                    self._last_scr_audit_tensors):
                # These are graph references only.  They allow the frozen
                # diagnostic to differentiate the detector's real training
                # score and SCR through the same upstream decoder tensor.
                self._last_scr_audit_tensors.update(
                    shared_hidden_states=head_inputs['hidden_states'],
                    matching_cls=matching_cls,
                    protected_mask=protected_mask,
                    base_pos_mask=sgr_matches['base_pos_mask'],
                    match_batch_idx=sgr_matches['match_batch_idx'],
                    match_student_idx=sgr_matches['match_student_idx'],
                    num_dn=matching_feats.new_tensor(float(num_dn)))
        return losses
