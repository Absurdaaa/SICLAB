from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def group_count(num_channels: int, max_groups: int = 32) -> int:
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return groups


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        factor = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -factor)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(group_count(channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x).reshape(b, c, h * w)
        q, k, v = self.qkv(x).chunk(3, dim=1)

        head_dim = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)

        attn = torch.softmax(torch.einsum("bncd,bnce->bnde", q, k) / math.sqrt(head_dim), dim=-1)
        out = torch.einsum("bnde,bnce->bncd", attn, v).reshape(b, c, h * w)
        out = self.proj(out).reshape(b, c, h, w)
        return out + residual


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (2, 2, 2),
        num_res_blocks: int = 4,
        time_emb_dim: int = 512,
        dropout: float = 0.0,
        attention_levels: tuple[int, ...] = (1, 2),
    ) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        channels = [base_channels]
        current_channels = base_channels
        for level, mult in enumerate(channel_multipliers):
            out_channels = base_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                block = nn.ModuleList(
                    [
                        ResBlock(current_channels, out_channels, time_emb_dim, dropout=dropout),
                        AttentionBlock(out_channels) if level in attention_levels else nn.Identity(),
                    ]
                )
                level_blocks.append(block)
                current_channels = out_channels
                channels.append(current_channels)
            self.down_blocks.append(level_blocks)
            if level != len(channel_multipliers) - 1:
                self.downsamples.append(Downsample(current_channels))
                channels.append(current_channels)

        self.mid_block1 = ResBlock(current_channels, current_channels, time_emb_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(current_channels)
        self.mid_block2 = ResBlock(current_channels, current_channels, time_emb_dim, dropout=dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_channels = base_channels * mult
            level_blocks = nn.ModuleList()
            blocks_in_level = num_res_blocks + 1
            for _ in range(blocks_in_level):
                skip_channels = channels.pop()
                block = nn.ModuleList(
                    [
                        ResBlock(
                            current_channels + skip_channels,
                            out_channels,
                            time_emb_dim,
                            dropout=dropout,
                        ),
                        AttentionBlock(out_channels) if level in attention_levels else nn.Identity(),
                    ]
                )
                level_blocks.append(block)
                current_channels = out_channels
            self.up_blocks.append(level_blocks)
            if level != 0:
                self.upsamples.append(Upsample(current_channels))

        self.out_norm = nn.GroupNorm(group_count(current_channels), current_channels)
        self.out_conv = nn.Conv2d(current_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(timesteps)
        h = self.init_conv(x)

        skips = [h]
        for level, blocks in enumerate(self.down_blocks):
            for res_block, attn_block in blocks:
                h = res_block(h, time_emb)
                h = attn_block(h)
                skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)
                skips.append(h)

        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)

        for level, blocks in enumerate(self.up_blocks):
            for res_block, attn_block in blocks:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = res_block(h, time_emb)
                h = attn_block(h)
            if level < len(self.upsamples):
                h = self.upsamples[level](h)

        return self.out_conv(F.silu(self.out_norm(h)))
