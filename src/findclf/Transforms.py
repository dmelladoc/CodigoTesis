from typing import Tuple

import torch
from torchvision.transforms import v2 as transforms


def _expand_dims(image: torch.Tensor) -> torch.Tensor:
    return image.unsqueeze(0)


def _permute(image: torch.Tensor) -> torch.Tensor:
    return image.permute(0, 3, 1, 2)


def get_transforms(
    input_shape: Tuple[int, int] = (256, 256), expand_dims: bool = True
) -> transforms.Compose:
    """
    Default transforms for our models.

    Args:
        input_shape (Tuple[int, int]): Desired input shape (height, width). Defaults to (256, 256).

    Returns:
        transforms.Compose: Composed transforms including dimension expansion, permutation, resizing, and type conversion.
    """
    transform_list = []
    # Add expand_dims transform if specified
    if expand_dims:
        transform_list += [transforms.Lambda(_expand_dims)]

    # Add permutation, resizing, and type conversion transforms
    transform_list += [
        transforms.Lambda(_permute),
        transforms.Resize(
            input_shape,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToDtype(torch.float32, scale=True),
    ]

    return transforms.Compose(transform_list)
