import types
import torch
import torch.nn as nn
from .model_utils import MaskLayer, Scale


def convert_to_teacher_model(model: nn.Module, skip_blocks: int = 0, prefix_only: bool = False) -> nn.Module:
    """
    Convert to teacher model.

    Args:
      model: nn.Module, original model.
      skip_blocks: int, number of blocks to skip from the output layer.
    """
    # Setup for masking layer.
    image_width = model.patch_embed.img_size[0]
    patch_size = model.patch_embed.patch_size[0]
    model.mask_layer = MaskLayer(image_width, patch_size)

    # Setup for output scaling layer.
    model.output_scaling = Scale(1)

    # Override forward function.
    if prefix_only:
        # NOTE: may not work for models with != 1 prefix token (SigLIP, DINOv2).
        assert skip_blocks == 0, "Require skip_blocks == 0 when prefix_only is True"

        def _forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            x = self.mask_layer(x, mask)
            x = self.forward_features(x)
            x = self.forward_head(x)
            x = self.output_scaling(x)
            return x

    else:
        # Prune blocks.
        if skip_blocks > 0:
            model.blocks = model.blocks[:-skip_blocks]

        def _forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            x = self.mask_layer(x, mask)
            x = self.forward_features(x)
            x = self.output_scaling(x)
            return x

    model.forward = types.MethodType(_forward, model)

    return model


def convert_to_mim_teacher_model(model: nn.Module) -> nn.Module:
    """
    Convert to MIM teacher model.

    Args:
      model: nn.Module, original model.
      skip_blocks: int, number of blocks to skip from the output layer.
    """
    # Setup for output scaling layer.
    model.output_scaling = Scale(1)

    # Override forward function.
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.output_scaling(x)
        return x

    model.forward = types.MethodType(_forward, model)

    return model
