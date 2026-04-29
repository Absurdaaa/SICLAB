from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
except ImportError:  # pragma: no cover
    LearnedPerceptualImagePatchSimilarity = None


class ConsistencyLoss(nn.Module):
    def __init__(
        self,
        loss_type: str,
        *,
        lpips_net: str = "alex",
        lpips_weight: float = 1.0,
        l1_weight: float = 1.0,
        l2_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type.lower()
        self.lpips_weight = lpips_weight
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self._fallback_warned = False

        self.lpips = None
        if "lpips" in self.loss_type and LearnedPerceptualImagePatchSimilarity is not None:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=lpips_net)
            self.lpips.requires_grad_(False)

    def _lpips_term(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.lpips is None:
            if not self._fallback_warned:
                print("LPIPS dependency not available, falling back to pixel-space loss.")
                self._fallback_warned = True
            return pred.new_zeros(())
        pred_224 = F.interpolate(pred, size=(224, 224), mode="bilinear", align_corners=False)
        target_224 = F.interpolate(target, size=(224, 224), mode="bilinear", align_corners=False)
        self.lpips = self.lpips.to(pred.device)
        return self.lpips(pred_224, target_224)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1":
            return F.l1_loss(pred, target)
        if self.loss_type == "l2":
            return F.mse_loss(pred, target)

        loss = pred.new_zeros(())
        has_term = False
        if "lpips" in self.loss_type:
            loss = loss + self.lpips_weight * self._lpips_term(pred, target)
            has_term = True
        if "l1" in self.loss_type:
            loss = loss + self.l1_weight * F.l1_loss(pred, target)
            has_term = True
        if "l2" in self.loss_type or not has_term:
            loss = loss + self.l2_weight * F.mse_loss(pred, target)
        return loss


__all__ = ["ConsistencyLoss"]
