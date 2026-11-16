"""Optimizer wrapper with isolated detector/GEO gradient clipping."""

from typing import List

import torch
from mmengine.optim import OptimWrapper
from mmengine.registry import OPTIM_WRAPPERS


@OPTIM_WRAPPERS.register_module()
class GeoIsolatedClipOptimWrapper(OptimWrapper):
    """Clip GEO-head and original-model parameters independently.

    The optimizer constructor preserves ``geo_group=True`` from paramwise_cfg
    on GEO parameter groups.  Clipping runs only at optimizer steps because it
    is implemented in ``step``, after gradient accumulation has completed.
    """

    def _clip_grad(self) -> None:
        detector: List[torch.Tensor] = []
        geo: List[torch.Tensor] = []
        for group in self.optimizer.param_groups:
            target = geo if bool(group.get('geo_group', False)) else detector
            target.extend(group['params'])
        detector = [p for p in detector if p.requires_grad and p.grad is not None]
        geo = [p for p in geo if p.requires_grad and p.grad is not None]
        kwargs = dict(self.clip_grad_kwargs)
        if detector:
            norm = self.clip_func(detector, **kwargs)
            if norm is not None:
                self.message_hub.update_scalar('train/detector_grad_norm',
                                               float(norm))
        if geo:
            norm = self.clip_func(geo, **kwargs)
            if norm is not None:
                self.message_hub.update_scalar('train/geo_grad_norm',
                                               float(norm))
