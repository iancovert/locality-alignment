import timm
import torch
import torch.nn as nn
from typing import Optional
from timm.models.vision_transformer import Block


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
