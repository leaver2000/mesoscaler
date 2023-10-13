# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
"""This file contains the Masked Autoencoder with VisionTransformer backbone, adapted from facebookresearch/mae_st."""
import os
from typing import TYPE_CHECKING, Callable, Literal, TypeAlias

import torch
import torch.nn as nn

from ..typing import StrPath
from .utils import Block, PatchEmbed

Loss: TypeAlias = torch.Tensor
Prediction: TypeAlias = torch.Tensor
Mask: TypeAlias = torch.Tensor
NormalizationStrategy: TypeAlias = Literal["layer", "group"]


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    if TYPE_CHECKING:
        __call__: Callable[..., tuple[Loss, Prediction, Mask]]

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio: float = 4.0,
        norm_pix_loss: bool = False,
        n_frames: int = 16,
        t_patch_size: int = 4,
        # patch_embed=video_vit.PatchEmbed,
        no_qkv_bias: bool = False,
        sep_pos_embed: bool = False,
        trunc_init: bool = False,
        cls_embed: bool = False,
        pred_t_dim: int = 8,
        norm: NormalizationStrategy = "layer",
        # **kwargs,
    ) -> None:
        super().__init__()

        self.config = {
            "img_size": img_size,
            "patch_size": patch_size,
            "in_chans": in_chans,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "norm_pix_loss": norm_pix_loss,
            "n_frames": n_frames,
            "t_patch_size": t_patch_size,
            "no_qkv_bias": no_qkv_bias,
            "sep_pos_embed": sep_pos_embed,
            "trunc_init": trunc_init,
            "cls_embed": cls_embed,
            "pred_t_dim": pred_t_dim,
            "decoder_embed_dim": decoder_embed_dim,
            "decoder_depth": decoder_depth,
            "decoder_num_heads": decoder_num_heads,
        }

        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        self.pred_t_dim = pred_t_dim
        self.t_pred_patch_size = t_patch_size * pred_t_dim // n_frames

        self.patch_embed = PatchEmbed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            n_frames,
            t_patch_size,
        )
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.num_channels = in_chans
        self.input_size = input_size

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(torch.zeros(1, input_size[1] * input_size[2], embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, input_size[0], embed_dim))
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim),
            )

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.decoder_pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], decoder_embed_dim)
            )
            self.decoder_pos_embed_temporal = nn.Parameter(torch.zeros(1, input_size[0], decoder_embed_dim))
            if self.cls_embed:
                self.decoder_pos_embed_class = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, decoder_embed_dim),
            )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    # norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            in_features=decoder_embed_dim,
            out_features=self.t_pred_patch_size * patch_size**2 * in_chans,
            bias=True,
        )

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        print("model initialized")

    def initialize_weights(self) -> None:
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def einsum(self, x: torch.Tensor, *, encode: bool) -> torch.Tensor:
        equation = "nctuhpwq->nthwupqc" if encode else "nthwupqc->nctuhpwq"
        return torch.einsum(equation, x)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (N, C, L, W)
        x: (N, L, patch_size**2 *C)
        """
        N, _, T, L, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        u = self.t_pred_patch_size

        assert (H == W) and (H % p == 0) and (T % u == 0)  # type: ignore
        h = w = H // p
        t = T // u

        x = imgs.reshape(shape=(N, self.num_channels, t, u, h, p, w, p))
        x = self.einsum(x, encode=True)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * self.num_channels))
        self.patch_info = (N, T, L, W, p, u, t, h, w)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, L, patch_size**2 *C)
        imgs: (N, 3, L, W)
        """
        N, T, L, W, p, u, t, h, w = self.patch_info

        x = x.reshape(shape=(N, t, h, w, u, p, p, self.num_channels))
        x = self.einsum(x, encode=False)
        imgs = x.reshape(shape=(N, self.num_channels, T, L, W))
        return imgs

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # un-shuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape

        x = x.reshape(N, T * L, C)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        x = x.view(N, -1, C)

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(1, self.input_size[0], 1) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        x = x.view([N, -1, C]) + pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        N = x.shape[0]
        T = self.patch_embed.t_grid_size
        H = W = self.patch_embed.grid_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([N, T * H * W, C])
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]))  # un-shuffle
        x = x_.view([N, T * H * W, C])
        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class.expand(decoder_pos_embed.shape[0], -1, -1),
                        decoder_pos_embed,
                    ],
                    1,
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        x = x + decoder_pos_embed

        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape  # type: ignore
        if requires_t_shape:
            x = x.view([N, T, H * W, C])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if requires_t_shape:
            x = x.view([N, T * H * W, -1])

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        imgs: [N, 3, T, L, W]
        pred: [N, t*h*w, u*p*p*3]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """
        _imgs = torch.index_select(
            imgs, 2, torch.linspace(0, imgs.shape[2] - 1, self.pred_t_dim).long().to(imgs.device)
        )
        target = self.patchify(_imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs: torch.Tensor, mask_ratio=0.75) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def to_disk(self, src: StrPath = "model.pt") -> None:
        src = os.path.abspath(src)
        os.makedirs(os.path.dirname(src), exist_ok=True)
        torch.save({"state": self.state_dict(), "config": self.config}, src)

    @classmethod
    def from_disk(cls, src: StrPath = "model.pt") -> "MaskedAutoencoderViT":
        ckpt = torch.load(src)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state"])
        return model


# def mae_vit_base_patch16(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=16,
#         embed_dim=768,
#         depth=12,
#         num_heads=12,
#         mlp_ratio=4,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs,
#     )
#     return model


# def mae_vit_large_patch16(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=16,
#         embed_dim=1024,
#         depth=24,
#         num_heads=16,
#         mlp_ratio=4,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs,
#     )
#     return model


# def mae_vit_huge_patch14(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=14,
#         embed_dim=1280,
#         depth=32,
#         num_heads=16,
#         mlp_ratio=4,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs,
#     )
#     return model
