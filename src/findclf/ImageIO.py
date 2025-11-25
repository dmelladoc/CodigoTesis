import os
from glob import glob
from typing import Sequence, Tuple

import cv2
import numpy as np
import pydicom as dcm
from pydicom.pixel_data_handlers import apply_voi_lut

from .Defaults import SUPPORTED_IMAGE_FORMATS


def normalize(image: np.ndarray) -> np.ndarray:
    return (image - image.min()) / (image.max() - image.min())


def load_dicom(path: str) -> np.ndarray:
    """
    Load and process a DICOM image file.

    Parameters
    ----------
    path : str
        Path to the DICOM file to load.

    Returns
    -------
    np.ndarray
        Normalized image array with pixel values in the range [0, 1].

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    TypeError
        If the image data type is not supported.
    """
    # check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found.")

    # Open DICOM file
    with dcm.dcmread(path) as ds:
        # apply VOI LUT if present
        image = apply_voi_lut(ds.pixel_array, ds)
        imtype = image.dtype

        # check if image is inverted
        if ds.PhotometricInterpretation == "MONOCHROME1":
            # if dtype is integer based:
            if imtype in [np.uint8, np.uint16, np.int16, np.int32]:
                image ^= np.iinfo(imtype).max
            # if dtype is float based:
            elif imtype in [np.float16, np.float32, np.float64]:
                image = np.max(image) - image
            else:
                raise TypeError(f"Unsupported image data type: {imtype}")

        # normalization
        image = normalize(image)
        return image


def load_image(path: str) -> np.ndarray:
    """
    Load image from file (png, jpg, etc.).

    Parameters
    ----------
    path : str
        Path to the image file to load.

    Returns
    -------
    np.ndarray
        Image array in RGB format.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found.")

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _applyCLAHE(
    image: np.ndarray, clipLimit: float, tile_size: Tuple[int, int]
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the input image.

    Arguments
    ---------
    image : np.ndarray
        Input grayscale image.
    clipLimit : float
        Threshold for contrast limiting.
    tile_size : Tuple[int, int]
        Size of the grid for histogram equalization.

    Returns
    -------
    np.ndarray
        Image after applying CLAHE.
    """

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tile_size)
    return clahe.apply(image)


def clahefusion(image: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """
    Apply CLAHE with multiple thresholds and fuse the results into a multi-channel image.

    Arguments
    ---------
    image : np.ndarray
        Input imae in uint8 format. If the image is not grayscale, it will be converted to grayscale.
    thresholds : Sequence[float]
        List of (ideally 2) CLAHE clip limit thresholds to apply.

    Returns
    -------
    np.ndarray
        Fused image with multiple channels, each corresponding to a CLAHE threshold.

    Raises
    ------
    TypeError
        If the input image is not of type uint8.

    """

    # Check if image dtype is uint8
    if image.dtype != np.uint8:
        raise TypeError("Input image must be of type uint8.")

    # check if image is grayscale
    if not len(image.shape) == 2:
        print("Warn: Converting to grayscale")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    bg = image == 0  # background mask
    list_of_images = [image.copy()]
    for thresh in thresholds:
        clahed_image = _applyCLAHE(image, clipLimit=thresh, tile_size=(8, 8))
        clahed_image[bg] = 0  # reapply background mask
        list_of_images.append(clahed_image)

    # fuse images to build fusion image, which each channel corresponding to a CLAHE threshold
    fused_image = cv2.merge(list_of_images)
    return fused_image


def get_ROIbox(image: np.ndarray) -> np.ndarray:
    """
    Extracts the bounding box of the region of interest (ROI) from the input image.

    Parameters
    ----------
    image : np.ndarray
        Input image from which to extract the ROI bounding box.
    Returns
    -------
    np.ndarray
        Bounding box of the ROI in the format [x_min, y_min, x_max, y_max].
    """
    # convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # apply otsu thresholding
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # find the largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    # get bounding box
    x, y, w, h = cv2.boundingRect(contour)  # x, y, width, height
    return np.array([x, y, x + w, y + h], dtype=np.int64)


def _check_image_format(path: str) -> bool:
    """
    Check if the file at a given path has a supported image format.
    """
    _, ext = os.path.splitext(path)
    return ext.lower() in SUPPORTED_IMAGE_FORMATS


def get_image_paths(input_paths: Sequence[str]) -> Sequence[str]:
    image_paths = []
    if len(input_paths) == 0:
        raise ValueError("No input paths provided.")

    # check if paths are files or directories
    for path in input_paths:
        # if path does not exist, then raise error
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input path {path} does not exist.")
        # if path is a file and is a supported image, add to list
        if os.path.isfile(path) and _check_image_format(path):
            image_paths.append(path)
        # if path is a directory, add all files in directory to list
        elif os.path.isdir(path):
            files = glob(os.path.join(path, "*"))
            # Check for all supported images
            files = [f for f in files if _check_image_format(f)]
            image_paths.extend(files)

    return image_paths


def save_image(image, path: str):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


def mix_heatmap(image, heatmap, alpha=0.5) -> np.ndarray:
    color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    weighted_heatmap = cv2.addWeighted(image, 1 - alpha, color_heatmap, alpha, 0)
    return weighted_heatmap
