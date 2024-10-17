import math
import torch
import numpy as np
from typing import Optional


class UniformMaskSampler:
    """
    Helper class to sample uniform masks.

    Args:
      mask_size: int, size of the mask.
      antithetical_sampling: bool, whether to use antithetical sampling.
      include_null: bool, whether to include the null mask for all samples.
    """

    def __init__(
        self,
        mask_size: int,
        antithetical_sampling: bool = True,
        include_null: bool = True,
    ) -> None:
        self.mask_size = mask_size
        self.include_null = include_null
        self.antithetical_sampling = antithetical_sampling
        self.rng = np.random.default_rng()

    def __call__(self, batch_size: int, seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate sample from mask distribution.

        Args:
          batch_size: int, number of inputs.
          seed: int, random seed.
        """
        # Setup.
        a = int(self.antithetical_sampling)
        b = int(self.include_null)
        if seed is None:
            rng = self.rng
        else:
            rng = np.random.default_rng(seed)

        # Sample masks.
        values = rng.uniform(size=(batch_size * (1 + a + b), self.mask_size + 1))
        masks = values[:, 1:] > values[:, :1]
        masks = torch.from_numpy(masks).float()

        # Modify masks.
        if self.antithetical_sampling:
            masks[1 :: (2 + b)] = 1 - masks[0 :: (2 + b)]
        if self.include_null:
            masks[(1 + a) :: (2 + a)] = 1

        return masks


class BlockwiseMaskSampler:
    """
    Helper class to sample blockwise masks.

    Args:
      mask_size: int, size of the mask.
      target_ratio: float, portion to be masked.
      antithetical_sampling: bool, whether to use antithetical sampling.
      include_null: bool, whether to include the null mask for all samples.
    """

    def __init__(
        self,
        mask_size: int,
        target_ratio: float = 0.4,
        antithetical_sampling: bool = True,
        include_null: bool = True,
    ) -> None:
        self.mask_size = mask_size
        self.target_ratio = target_ratio
        self.antithetical_sampling = antithetical_sampling
        self.include_null = include_null

        # Determine mask side length.
        self.mask_side = int(math.sqrt(self.mask_size))
        assert self.mask_side * self.mask_side == self.mask_size, "Mask size must be square"

    def __call__(self, batch_size: int, seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate sample from mask distribution.

        Args:
          batch_size: int, number of inputs.
          seed: int, random seed.
        """
        # Setup.
        rng = np.random.default_rng(seed)
        a = int(self.antithetical_sampling)
        b = int(self.include_null)
        masks = torch.zeros(batch_size * (1 + a + b), self.mask_size)
        min_block_size = 16
        max_block_size = int(self.target_ratio * self.mask_size)

        # Generate masks.
        for i in range(0, batch_size * (1 + a + b), 1 + a + b):
            mask = masks[i].view(self.mask_side, self.mask_side)
            while mask.mean() < self.target_ratio:
                # Block shape.
                block_size = rng.integers(min_block_size, max_block_size)
                aspect_ratio = rng.uniform(0.3, 1 / 0.3)
                width = min(int(math.sqrt(block_size / aspect_ratio)), self.mask_side)
                height = min(int(block_size / width), self.mask_side)

                # Block position.
                x_start = rng.integers(0, self.mask_side - width + 1)
                y_start = rng.integers(0, self.mask_side - height + 1)
                mask[x_start : x_start + width, y_start : y_start + height] = 1

        # Antithetical sampling.
        if self.antithetical_sampling:
            for i in range(1, batch_size * (1 + a + b), 2 + a):
                masks[i] = 1 - masks[i - 1]

        return masks


class BernoulliMaskSampler:
    """
    Helper class to sample Bernoulli masks.

    Args:
      mask_size: int, size of the mask.
      mask_ratio: float, portion to be masked.
      antithetical_sampling: bool, whether to use antithetical sampling.
      include_null: bool, whether to include the null mask for all samples.
    """

    def __init__(
        self,
        mask_size: int,
        mask_ratio: float = 0.5,
        antithetical_sampling: bool = True,
        include_null: bool = True,
    ) -> None:
        self.mask_size = mask_size
        self.mask_ratio = mask_ratio
        self.antithetical_sampling = antithetical_sampling
        self.include_null = include_null

    def __call__(self, batch_size: int, seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate sample from mask distribution.

        Args:
          batch_size: int, number of inputs.
          seed: int, random seed.
        """
        # Setup.
        rng = np.random.default_rng(seed)
        a = int(self.antithetical_sampling)
        b = int(self.include_null)

        # Generate masks.
        masks = rng.uniform(size=(batch_size * (1 + a + b), self.mask_size)) < self.mask_ratio
        masks = torch.from_numpy(masks).float()

        # Modify masks.
        if self.antithetical_sampling:
            masks[1 :: (2 + b)] = 1 - masks[0 :: (2 + b)]
        if self.include_null:
            masks[(1 + a) :: (2 + a)] = 1

        return masks
