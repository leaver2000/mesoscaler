# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from __future__ import annotations

import collections.abc
import itertools
import logging
import logging.config
from collections import deque
from typing import Callable, TypeVar

import torch
import torch.nn as nn
from timm.models.vision_transformer import DropPath, Mlp

from ..typing import Batch, Channel, Length, N, Number, TensorF32, Time, Width

logger = logging.getLogger(__name__)


T = TypeVar("T")


def to_pair(x: T | tuple[T, T]) -> tuple[T, T]:
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(itertools.repeat(x, 2))


class PatchEmbed(nn.Module):
    """Image to Patch Embedding

    Patches the [T, L, W] input image into [T, H//patch_size, W//patch_size, patch_size**2] patches
    """

    proj: Callable[..., torch.Tensor]

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        # temporal related:
        frames: int = 32,
        t_patch_size: int = 4,
    ):
        super().__init__()
        img_size = to_pair(img_size)
        patch_size = to_pair(patch_size)

        assert img_size[1] % patch_size[1] == 0
        assert img_size[0] % patch_size[0] == 0
        assert frames % t_patch_size == 0
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (frames // t_patch_size)
        self.input_size = (
            frames // t_patch_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        # print(f"img_size {img_size} patch_size {patch_size} frames {frames} t_patch_size {t_patch_size}")
        print(
            # f"img_size {img_size} patch_size {patch_size} frames {frames} t_patch_size {t_patch_size}",
            f"[input]\ntorch.Size([B, {in_chans}, {frames}, {img_size[0]}, {img_size[1]}])",
            f"[patch]\ntorch.Size([B, {in_chans}, {frames // t_patch_size}, {img_size[0] // patch_size[0]}, {img_size[1] // patch_size[1]}])",
            sep="\n",
        )

        self.frames = frames
        self.num_patches = num_patches

        self.patch_size = patch_size
        self.img_size = img_size

        self.grid_size = img_size[0] // patch_size[0]

        self.t_patch_size = t_patch_size
        self.t_grid_size = frames // t_patch_size

        kernel_size = (t_patch_size,) + patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size)
        self.shape = (self.frames, *self.img_size)

    def forward(self, x: TensorF32[Batch, Channel, Time, Length, Width]) -> TensorF32[N, Time, N, Channel]:
        B, C, T, L, W = x.shape
        if not (T, L, W) == self.tlw:
            logger.error(f"Input image size (B, C, {T}, {L}, {W}) doesn't match model ({self.shape}).")
            raise ValueError(f"Input image size (B, C, {T}, {L}, {W}) doesn't match model ({self.shape}).")
        x = self.proj(x).flatten(3)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]
        return x


class Attention(nn.Module):
    q: Callable[..., torch.Tensor]
    k: Callable[..., torch.Tensor]
    v: Callable[..., torch.Tensor]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: int | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        input_size: tuple[int, int, int] = (4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        assert attn_drop == 0.0  # do not use
        assert input_size[1] == input_size[2]

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x


class Block(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    norm1: Callable[..., torch.Tensor]
    attn: Callable[..., torch.Tensor]
    drop_path: Callable[..., torch.Tensor]
    norm2: Callable[..., torch.Tensor]
    mlp: Callable[..., torch.Tensor]

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self, x: TensorF32[Batch, Time, Channel, Length, Width]
    ) -> TensorF32[Batch, Time, Channel, Length, Width]:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# =====================================================================================================================
class MetricLogger(logging.Logger):
    def __init__(self):
        super().__init__(__name__, logging.INFO)

        self.meters: dict[str, MetricValue] = {}
        self.info("Initialized logger")

    def add_meter(self, name: str, meter: MetricValue) -> None:
        self.meters[name] = meter

    def update(self, **kwargs: Number | torch.Tensor | None) -> None:
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            v = v
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def write(self, epoch: str):
        data = {k: str(v) for k, v in self.meters.items()}
        self.info(f"epoch[{epoch}] {data}")


class MetricValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, *, maxlen: int = 20, fmt: str = "{median:.4f} ({global_avg:.4f})") -> None:
        self.deque = deque[Number](maxlen=maxlen)
        self.total = 0.0
        self.count = 1
        self.fmt = fmt

    def update(self, value: Number, n: int = 1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def to_tensor(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.tensor(list(self.deque), dtype=dtype)

    @property
    def median(self) -> Number:
        return self.to_tensor().median().item()

    @property
    def avg(self) -> Number:
        return self.to_tensor(dtype=torch.float32).mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / self.count

    @property
    def max(self) -> Number:
        return max(self.deque)

    @property
    def value(self) -> Number:
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )
