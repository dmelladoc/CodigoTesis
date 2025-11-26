from itertools import product
from typing import Sequence, Tuple

import numpy as np
from cv2 import INTER_LINEAR, resize
from scipy.signal import convolve2d


def crop_window(
    image: np.ndarray,
    center: Tuple[int, int],
    window_size: int,
    expand_dims: bool = False,
) -> np.ndarray:
    """
    Crops a window centered at a given point of the image.

    Parameters:
        image (np.ndarray): The input image from which to crop.
        center (Tuple[int, int]): The (y, x) coordinates of the center point.
        window_size (int): The size of the square window to crop.
        expand_dims (bool): Whether to expand dimensions of the output array. Default is False.

    Returns:
        np.ndarray: The cropped image window.
    """
    half = window_size // 2
    y0, x0, y1, x1 = (
        center[0] - half,
        center[1] - half,
        center[0] + half,
        center[1] + half,
    )
    crop = image[y0:y1, x0:x1]
    if expand_dims:
        crop = np.expand_dims(crop, axis=0)
    return crop


def get_windows(
    image: np.ndarray, window_size: int, stride: int
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Crops overlapping windows from the input image.

    Parameters:
        image (np.ndarray): The input image from which to crop windows.
        window_size (int): The size of the square windows to crop.
        stride (int): The stride between consecutive windows.
    Returns:
        Tuple[np.ndarray, Tuple[int, int]]: A tuple containing:
            - An array of cropped image windows.
            - A tuple of arrays representing the x and y coordinates of the window centers.

    """

    height, width = image.shape[:2]  # get height and width of the image
    half = window_size // 2  # calculate half window size
    padding = ((half, half)), ((half, half))  # define padding for height and width
    if len(image.shape) == 3:
        padding += ((0, 0),)  # no padding for channels

    padded_image = np.pad(
        image, padding, mode="constant", constant_values=0
    )  # pad the image with zeros
    pad_w, pad_h = padded_image.shape[:2]  # get new height and width after padding

    # obtain coordinates of windows
    dx = np.arange(half, pad_w - half, stride)
    dy = np.arange(half, pad_h - half, stride)
    centers = list(product(dx, dy))

    # obtain the set of windows from the image
    windows = np.array(
        [crop_window(padded_image, center, window_size) for center in centers]
    )
    return windows, (dx, dy)


# Kernel ops
def _boxIOU(boxA, boxB):
    x0 = max(boxA[0], boxB[0])
    y0 = max(boxA[1], boxB[1])
    x1 = min(boxA[2], boxB[2])
    y1 = min(boxA[3], boxB[3])

    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area1 = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area2 = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return intersection / float(area1 + area2 - intersection)


def _IOU_from_center(px, py, half: int = 1):
    box1 = np.array([-half, -half, half, half])
    box2 = np.array([px - half, py - half, px + half, py + half])
    return _boxIOU(box1, box2)


def create_kernel(window_size: int, stride: int, norm: bool = True):
    half = window_size // 2
    range_window = np.arange(-half, half + 1, stride)
    xx, yy = np.meshgrid(range_window, range_window)
    kernel = np.vectorize(_IOU_from_center)(xx, yy, half=half)
    if not norm:
        return kernel
    return kernel / np.sum(kernel)


def apply_overlap_kernel(
    predmap: np.ndarray,
    kernel: np.ndarray,
    image_shape: Sequence[int],
    window_size: int,
) -> np.ndarray:
    """Applays overlap kernel to prediction map.

    Args:
        predmap (np.ndarray): Prediction map to be convolved.
        kernel (np.ndarray): Overlap kernel.
        image_shape (Sequence[int]): Original image shape.
        window_size (int): Size of the window used.
    Returns:
        np.ndarray: Resized prediction map after applying the kernel.
    """
    half = window_size // 2
    out_shape = image_shape + np.array([window_size, window_size])
    convmap = convolve2d(predmap, kernel, mode="full")
    resized = resize(convmap, out_shape, interpolation=INTER_LINEAR)
    return resized[half:-half, half:-half]


def resize_map_to_image(
    predmap: np.ndarray,
    image_shape: Sequence[int],
    window_size: int,
) -> np.ndarray:
    half = window_size // 2
    out_shape = image_shape + np.array([window_size, window_size])
    resized = resize(predmap, out_shape, interpolation=INTER_LINEAR)
    return resized[half:-half, half:-half]


def batcher(iterable, batch_size: int = 1):
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        yield iterable[ndx : min(ndx + batch_size, length)]
