# Modified from https://github.com/cambrian-mllm/cambrian/

import os
import math
import torch
import torch.nn as nn
import logging
from functools import partial, reduce
from operator import mul
from typing import Optional

import timm
from timm.models.vision_transformer import VisionTransformer
from timm.layers.helpers import to_2tuple
from timm.models.layers import PatchEmbed
from timm.models.registry import register_model


class VisionTransformerMoCo(VisionTransformer):
    """Vision transformer with sin-cos position embeddings and custom initialization."""

    def __init__(self, stop_grad_conv1: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding.
        self.build_2d_sincos_position_embedding()

        # Weight initialization.
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "qkv" in name:
                    # Treat the weights of Q, K, V separately.
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # Xavier uniform initialization.
            val = math.sqrt(6.0 / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature: float = 10000.0) -> None:
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert self.embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
            dim=1,
        )[None, :, :]

        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False


class ConvStem(nn.Module):
    """
    Convolutional patch embedding module from https://arxiv.org/abs/2106.14881
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
    ):
        super().__init__()

        assert patch_size == 16, "ConvStem only supports patch size of 16"
        assert embed_dim % 8 == 0, "Embed dimension must be divisible by 8 for ConvStem"

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # Build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = in_chans, embed_dim // 8
        for _ in range(4):
            stem.append(
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def _create_moco_vit(variant: str, pretrained: bool = False, **kwargs) -> VisionTransformerMoCo:
    img_size = int(variant.split("_")[-1])
    pretrained_cfg = {
        "architecture": variant,
        "input_size": (3, img_size, img_size),
        "fixed_input_size": True,
        "interpolation": "bicubic",
        "crop_pct": 0.875,
        "crop_mode": "center",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "num_classes": 1000,
        "pool_size": None,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
    }

    if "pretrained_cfg" in kwargs:
        if kwargs["pretrained_cfg"] is None or kwargs["pretrained_cfg"] == {}:
            kwargs["pretrained_cfg"] = pretrained_cfg
        else:
            raise ValueError("pretrained_cfg is not empty")

    model = timm.models._builder.build_model_with_cfg(
        VisionTransformerMoCo,
        variant,
        pretrained=False,
        **kwargs,
    )

    if pretrained:
        if variant.startswith("vit_base"):
            # Load checkpoint.
            cache_dir = torch.hub.get_dir()
            local_path = os.path.join(cache_dir, "checkpoints", "vit-b-300ep.pth.tar")

            url = "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
            if os.path.exists(local_path):
                logging.info(f"Loading moco/vit-b-300ep.pth.tar from local path: {local_path}")
                ckpt = torch.load(local_path, map_location=torch.device("cpu"))
            else:
                logging.info(f"Downloading moco/vit-b-300ep.pth.tar from url: {url}")
                ckpt = torch.hub.load_state_dict_from_url(url, map_location=torch.device("cpu"))
            pretrained_dict = ckpt["state_dict"]

            # Fix state dict keys.
            pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
            pretrained_dict = {
                k.replace("base_encoder.", ""): v for k, v in pretrained_dict.items() if k.startswith("base_encoder.")
            }

            non_matching_keys = model.load_state_dict(pretrained_dict, strict=False)
            logging.info(f"Non-matching keys: {non_matching_keys}")
            print(f"Non-matching keys: {non_matching_keys}")
        else:
            raise ValueError(f"Pretrained weights not available for {variant}")

    return model


@register_model
def vit_base_patch16_mocov3_224(pretrained: bool = False, **kwargs) -> VisionTransformerMoCo:
    """Moco-v3 ViT-B/16 based on https://arxiv.org/abs/2104.02057."""
    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    model = _create_moco_vit(
        "vit_base_patch16_mocov3_224",
        pretrained=pretrained,
        **dict(model_args, **kwargs),
    )
    return model
