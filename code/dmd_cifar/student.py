from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ddpm_cifar.model import AttentionBlock, ResBlock, SinusoidalTimeEmbedding, group_count


class OneStepGenerator(nn.Module):
    """A small pixel-space generator used for one-step distillation."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 2, 2),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        attention_levels: tuple[int, ...] = (2,),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.emb_dim = base_channels * 4
        self.sigma_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_channels = base_channels
        skip_channels: list[int] = [current_channels]

        for level, mult in enumerate(channel_multipliers):
            out_channels = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    nn.ModuleList(
                        [
                            ResBlock(current_channels, out_channels, time_emb_dim=self.emb_dim, dropout=dropout),
                            AttentionBlock(out_channels) if level in attention_levels else nn.Identity(),
                        ]
                    )
                )
                current_channels = out_channels
                skip_channels.append(current_channels)
            self.down_blocks.append(blocks)
            if level != len(channel_multipliers) - 1:
                self.downsamples.append(
                    nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding=1)
                )
                skip_channels.append(current_channels)

        self.mid = nn.Sequential(
            nn.GroupNorm(group_count(current_channels), current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
        )

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_channels = base_channels * mult
            blocks = nn.ModuleList()
            blocks_in_level = num_res_blocks + 1
            for _ in range(blocks_in_level):
                skip_ch = skip_channels.pop()
                blocks.append(
                    nn.ModuleList(
                        [
                            ResBlock(current_channels + skip_ch, out_channels, time_emb_dim=self.emb_dim, dropout=dropout),
                            AttentionBlock(out_channels) if level in attention_levels else nn.Identity(),
                        ]
                    )
                )
                current_channels = out_channels
            self.up_blocks.append(blocks)
            if level != 0:
                self.upsamples.append(nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1))

        self.out_norm = nn.GroupNorm(group_count(current_channels), current_channels)
        self.out_conv = nn.Conv2d(current_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.reshape(-1).to(z.device, z.dtype)
        sigma_emb = self.sigma_embed(torch.log(sigma.clamp_min(1e-4)))
        h = self.input_conv(z)
        skips = [h]

        for level, blocks in enumerate(self.down_blocks):
            for res_block, attn_block in blocks:
                h = res_block(h, sigma_emb)
                h = attn_block(h)
                skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)
                skips.append(h)

        h = self.mid(h)

        for level, blocks in enumerate(self.up_blocks):
            for res_block, attn_block in blocks:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = res_block(h, sigma_emb)
                h = attn_block(h)
            if level < len(self.upsamples):
                h = F.interpolate(h, scale_factor=2.0, mode="nearest")
                h = self.upsamples[level](h)

        return torch.tanh(self.out_conv(F.silu(self.out_norm(h))))

__all__ = ["OneStepGenerator"]
