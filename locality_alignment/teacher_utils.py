import types
import torch
import torch.nn as nn
from .model_utils import Scale


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
