import timm
import torch
import torch.nn as nn
from typing import Optional
from timm.models.vision_transformer import Block


class MaskLayer(nn.Module):
    """
    Mask layer for images.

    Args:
      image_width: int, width of the image.
      patch_size: int, width of each image patch.
    """

    def __init__(self, image_width: int, patch_size: int) -> None:
        super(MaskLayer, self).__init__()
        self.mask_width = image_width // patch_size
        self.upsample_factor = patch_size

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.view(-1, 1, self.mask_width, self.mask_width)
        mask = nn.functional.interpolate(mask, scale_factor=self.upsample_factor)
        if mask.shape[2] != x.shape[2] or mask.shape[3] != x.shape[3]:
            # Zero-pad for irregular patch/image sizes (occurs for vit_so400m_patch14_siglip_384).
            height_pad = x.shape[2] - mask.shape[2]
            width_pad = x.shape[3] - mask.shape[3]
            mask = nn.functional.pad(mask, (0, width_pad, 0, height_pad))
        return x * mask


class Scale(nn.Module):
    """Scale output by a constant factor."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class MaskEmbedDecoder(nn.Module):
    """Decoder transformer module for MaskEmbed."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_patch_tokens: int,
        num_prefix_tokens: int,
        output_size: int,
        num_layers: int,
        prefix_only: bool = False,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        mlp_layer: nn.Module = timm.layers.Mlp,
    ) -> None:
        super().__init__()

        # Positional embeddings.
        self.pos_embed = nn.Parameter(torch.randn(1, num_patch_tokens + num_prefix_tokens, embed_dim) * 0.02)

        # Prefix tokens.
        self.prefix_only = prefix_only
        self.num_prefix_tokens = num_prefix_tokens

        # Transformer blocks.
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for _ in range(num_layers)
            ]
        )

        # Output normalization.
        self.norm = nn.LayerNorm(embed_dim)

        # Output head.
        self.head = nn.Linear(embed_dim, output_size)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional embeddings.
        return x + self.pos_embed

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Apply masking.
        if self.num_prefix_tokens > 0:
            # Pad with 0 to mask prefix tokens.
            mask = nn.functional.pad(mask, (self.num_prefix_tokens, 0), value=0)
        x = x * mask.unsqueeze(-1)

        # Generate predictions.
        x = self._pos_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)
        if self.prefix_only:
            x = x[:, : self.num_prefix_tokens]
        return x


class MaskEmbedStudent(nn.Module):
    """Wrapper around student encoder and decoder."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Encoder predictions.
        preds = self.encoder(x)

        # Decoder reconstruction.
        num_masks = len(mask) // len(preds)
        preds_repeat = preds.repeat_interleave(num_masks, 0)
        return self.decoder(preds_repeat, mask)


class TransformerPooling(nn.Module):
    """Small output head transformer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_tokens: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        # Positional embeddings.
        self.pos_embed = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.02)

        # Transformer blocks (parameters set to match ViT-B architecture).
        self.blocks = nn.Sequential(
            *[
                timm.models.vision_transformer.Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    qkv_bias=False,
                    qk_norm=False,
                    proj_drop=0,
                    attn_drop=0,
                    drop_path=0,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU,
                    mlp_layer=timm.layers.Mlp,
                )
                for _ in range(num_layers)
            ]
        )

        # Output normalization.
        self.norm = nn.LayerNorm(embed_dim)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply positional embeddings and transformer blocks.
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return x
