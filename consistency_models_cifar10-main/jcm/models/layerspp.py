# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining NCSN++.
"""
from typing import Any, Optional, Tuple
from . import layers
from . import up_or_down_sampling
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    embedding_size: int = 256
    scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        W = self.param(
            "W", jax.nn.initializers.normal(stddev=self.scale), (self.embedding_size,)
        )
        W = jax.lax.stop_gradient(W)
        x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class Combine(nn.Module):
    """Combine information from skip connections."""

    method: str = "cat"

    @nn.compact
    def __call__(self, x, y):
        h = conv1x1(x, y.shape[-1])
        if self.method == "cat":
            return jnp.concatenate([h, y], axis=-1)
        elif self.method == "sum":
            return h + y
        else:
            raise ValueError(f"Method {self.method} not recognized.")


class AdaGN(nn.Module):
    """Adaptive GroupNorm with scale/shift conditioning from class embedding.

    The scale and shift are predicted from a conditioning embedding and
    applied after GroupNorm.
    """

    out_ch: int
    num_groups: int = 32

    @nn.compact
    def __call__(self, x, cond):
        h = nn.GroupNorm(num_groups=min(x.shape[-1] // 4, self.num_groups))(x)
        if cond is None:
            return h

        scale_shift = nn.Dense(
            2 * self.out_ch,
            kernel_init=nn.initializers.zeros,
        )(nn.swish(cond))
        scale, shift = jnp.split(scale_shift, 2, axis=-1)
        h = h * (1.0 + scale[:, None, None, :]) + shift[:, None, None, :]
        return h


class CrossAttention(nn.Module):
    """Cross-attention from spatial image features to conditioning tokens."""

    out_ch: int
    num_heads: int = 4
    num_tokens: int = 4
    init_scale: float = 0.0
    skip_rescale: bool = True

    @nn.compact
    def __call__(self, x, cond, train=True):
        if cond is None:
            return x

        b, h, w, c = x.shape
        h_in = nn.GroupNorm(num_groups=min(c // 4, 32))(x)
        q = jnp.reshape(h_in, (b, h * w, c))
        if cond.ndim == 2:
            kv = nn.Dense(
                self.num_tokens * c,
                kernel_init=default_init(),
            )(nn.swish(cond))
            kv = jnp.reshape(kv, (b, self.num_tokens, c))
        elif cond.ndim == 3:
            kv = cond
            if kv.shape[-1] != c:
                kv = nn.Dense(c, kernel_init=default_init())(kv)
        else:
            raise ValueError(f"Unsupported conditioning rank {cond.ndim}.")

        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=c,
            out_features=self.out_ch,
            kernel_init=default_init(),
            out_kernel_init=default_init(self.init_scale),
            use_bias=True,
            dropout_rate=0.0,
        )
        h_out = attn(q, kv, deterministic=not train)
        h_out = jnp.reshape(h_out, (b, h, w, self.out_ch))
        if self.out_ch != c:
            x = conv1x1(x, self.out_ch)
        if not self.skip_rescale:
            return x + h_out
        return (x + h_out) / np.sqrt(2.0)


class AttnBlockpp(nn.Module):
    """Channel-wise self-attention block. Modified from DDPM."""

    skip_rescale: bool = False
    init_scale: float = 0.0

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        h = nn.GroupNorm(num_groups=min(x.shape[-1] // 4, 32))(x)
        q = NIN(C)(h)
        k = NIN(C)(h)
        v = NIN(C)(h)

        w = jnp.einsum("bhwc,bHWc->bhwHW", q, k) * (int(C) ** (-0.5))
        w = jnp.reshape(w, (B, H, W, H * W))
        w = jax.nn.softmax(w, axis=-1)
        w = jnp.reshape(w, (B, H, W, H, W))
        h = jnp.einsum("bhwHW,bHWc->bhwc", w, v)
        h = NIN(C, init_scale=self.init_scale)(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


class Upsample(nn.Module):
    out_ch: Optional[int] = None
    with_conv: bool = False
    fir: bool = False
    fir_kernel: Tuple[int] = (1, 3, 3, 1)

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        out_ch = self.out_ch if self.out_ch else C
        if not self.fir:
            h = jax.image.resize(x, (x.shape[0], H * 2, W * 2, C), "nearest")
            if self.with_conv:
                h = conv3x3(h, out_ch)
        else:
            if not self.with_conv:
                h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.Conv2d(
                    out_ch,
                    kernel=3,
                    up=True,
                    resample_kernel=self.fir_kernel,
                    use_bias=True,
                    kernel_init=default_init(),
                )(x)

        assert h.shape == (B, 2 * H, 2 * W, out_ch)
        return h


class Downsample(nn.Module):
    out_ch: Optional[int] = None
    with_conv: bool = False
    fir: bool = False
    fir_kernel: Tuple[int] = (1, 3, 3, 1)

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        out_ch = self.out_ch if self.out_ch else C
        if not self.fir:
            if self.with_conv:
                x = conv3x3(x, out_ch, stride=2)
            else:
                x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        else:
            if not self.with_conv:
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                x = up_or_down_sampling.Conv2d(
                    out_ch,
                    kernel=3,
                    down=True,
                    resample_kernel=self.fir_kernel,
                    use_bias=True,
                    kernel_init=default_init(),
                )(x)

        assert x.shape == (B, H // 2, W // 2, out_ch)
        return x


class ResnetBlockDDPMpp(nn.Module):
    """ResBlock adapted from DDPM."""

    act: Any
    out_ch: Optional[int] = None
    conv_shortcut: bool = False
    dropout: float = 0.1
    skip_rescale: bool = False
    init_scale: float = 0.0
    conditioning_type: str = "adagn"
    num_heads: int = 4
    num_tokens: int = 4

    @nn.compact
    def __call__(self, x, temb=None, class_emb=None, train=True):
        B, H, W, C = x.shape
        out_ch = self.out_ch if self.out_ch else C
        conditioning_type = self.conditioning_type.lower()
        if conditioning_type == "adagn":
            h = AdaGN(out_ch=out_ch)(x, class_emb)
            h = self.act(h)
        else:
            h = self.act(nn.GroupNorm(num_groups=min(x.shape[-1] // 4, 32))(x))
        h = conv3x3(h, out_ch)

        # Keep the original time-conditioning path so pretrained checkpoints
        # can still be reused.
        if temb is not None:
            h += nn.Dense(out_ch, kernel_init=default_init())(self.act(temb))[
                :, None, None, :
            ]

        if conditioning_type == "cross_attn" and class_emb is not None:
            h = CrossAttention(
                out_ch=out_ch,
                num_heads=self.num_heads,
                num_tokens=self.num_tokens,
                init_scale=self.init_scale,
                skip_rescale=False,
            )(h, class_emb, train=train)
            h = self.act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
        elif conditioning_type == "adagn":
            h = AdaGN(out_ch=out_ch)(h, class_emb)
            h = self.act(h)
        else:
            h = self.act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
        h = nn.Dropout(self.dropout)(h, deterministic=not train)
        h = conv3x3(h, out_ch, init_scale=self.init_scale)
        if C != out_ch:
            if self.conv_shortcut:
                x = conv3x3(x, out_ch)
            else:
                x = NIN(out_ch)(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


class ResnetBlockBigGANpp(nn.Module):
    """ResBlock adapted from BigGAN."""

    act: Any
    up: bool = False
    down: bool = False
    out_ch: Optional[int] = None
    dropout: float = 0.1
    fir: bool = False
    fir_kernel: Tuple[int] = (1, 3, 3, 1)
    skip_rescale: bool = True
    init_scale: float = 0.0
    conditioning_type: str = "adagn"
    num_heads: int = 4
    num_tokens: int = 4

    @nn.compact
    def __call__(self, x, temb=None, class_emb=None, train=True):
        B, H, W, C = x.shape
        out_ch = self.out_ch if self.out_ch else C
        conditioning_type = self.conditioning_type.lower()
        if conditioning_type == "adagn":
            h = AdaGN(out_ch=out_ch)(x, class_emb)
            h = self.act(h)
        else:
            h = self.act(nn.GroupNorm(num_groups=min(x.shape[-1] // 4, 32))(x))

        if self.up:
            if self.fir:
                h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

        h = conv3x3(h, out_ch)

        # Preserve time conditioning from the original architecture.
        if temb is not None:
            h += nn.Dense(out_ch, kernel_init=default_init())(self.act(temb))[
                :, None, None, :
            ]

        if conditioning_type == "cross_attn" and class_emb is not None:
            h = CrossAttention(
                out_ch=out_ch,
                num_heads=self.num_heads,
                num_tokens=self.num_tokens,
                init_scale=self.init_scale,
                skip_rescale=False,
            )(h, class_emb, train=train)
            h = self.act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
        elif conditioning_type == "adagn":
            h = AdaGN(out_ch=out_ch)(h, class_emb)
            h = self.act(h)
        else:
            h = self.act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
        h = nn.Dropout(self.dropout)(h, deterministic=not train)
        h = conv3x3(h, out_ch, init_scale=self.init_scale)
        if C != out_ch or self.up or self.down:
            x = conv1x1(x, out_ch)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)
