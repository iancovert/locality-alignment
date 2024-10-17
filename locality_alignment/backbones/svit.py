import torch
from torch import nn
from typing import Optional
from functools import partial

import timm
from timm.layers import Mlp, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import LayerScale, VisionTransformer


class StableAttention(nn.Module):
    """Stable self-attention with logit soft-capping."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        max_attn_val: float = 30.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.max_attn_val = max_attn_val

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Calculate attention with logit soft-capping.
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.max_attn_val * torch.tanh(attn / self.max_attn_val)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class StableBlock(nn.Module):
    """Transformer block with stable self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        max_attn_val: float = 30.0,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = StableAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            max_attn_val=max_attn_val,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def create_svit(model_name: str, pretrained: bool = False, **kwargs) -> VisionTransformer:
    block_fn = (
        partial(StableBlock, max_attn_val=kwargs.pop("max_attn_val")) if "max_attn_val" in kwargs else StableBlock
    )
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        block_fn=block_fn,
        **kwargs,
    )
    return model


@register_model
def svit_tiny_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    return create_svit("vit_tiny_patch16_224", pretrained, **kwargs)


@register_model
def svit_small_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    return create_svit("vit_small_patch16_224", pretrained, **kwargs)


@register_model
def svit_base_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    return create_svit("vit_base_patch16_224", pretrained, **kwargs)


@register_model
def svit_large_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    return create_svit("vit_large_patch16_224", pretrained, **kwargs)


@register_model
def svit_base_patch16_mocov3_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    return create_svit("vit_base_patch16_mocov3_224", pretrained, **kwargs)


@register_model
def svit_base_patch16_clip_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    return create_svit("vit_base_patch16_clip_224", pretrained, **kwargs)


@register_model
def svit_base_patch16_clip_quickgelu_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    return create_svit("vit_base_patch16_clip_quickgelu_224", pretrained, **kwargs)


@register_model
def svit_large_patch14_clip_quickgelu_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    return create_svit("vit_large_patch14_clip_quickgelu_224", pretrained, **kwargs)


@register_model
def svit_large_patch14_clip_quickgelu_336(pretrained: bool = False, **kwargs) -> VisionTransformer:
    return create_svit("vit_large_patch14_clip_quickgelu_336", pretrained, **kwargs)


@register_model
def svit_base_patch14_dinov2(pretrained: bool = False, **kwargs) -> VisionTransformer:
    return create_svit("vit_base_patch14_dinov2", pretrained, **kwargs)


@register_model
def svit_base_patch14_reg4_dinov2(pretrained: bool = False, **kwargs) -> VisionTransformer:
    return create_svit("vit_base_patch14_reg4_dinov2", pretrained, **kwargs)


@register_model
def svit_base_patch16_siglip_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    return create_svit("vit_base_patch16_siglip_224", pretrained, **kwargs)


@register_model
def svit_so400m_patch14_siglip_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    return create_svit("vit_so400m_patch14_siglip_224", pretrained, **kwargs)


@register_model
def svit_so400m_patch14_siglip_384(pretrained: bool = False, **kwargs) -> VisionTransformer:
    return create_svit("vit_so400m_patch14_siglip_384", pretrained, **kwargs)
