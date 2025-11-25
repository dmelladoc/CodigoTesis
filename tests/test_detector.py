from argparse import ArgumentParser
from os import makedirs
from os.path import isdir, isfile, splitext
from os.path import join as path_join

import cv2
import numpy as np
import torch

from correlax.ImageIO import clahefusion, get_ROIbox, load_dicom
from findclf.ModelOps import Detector


def main(args):
    # check if file exists
    if not isfile(args.input):
        raise FileNotFoundError(f"Input file not found at {args.input}")

    # open DICOM file and extract the image
    if not splitext(args.input)[1].lower() in [".dcm", ".dicom"]:
        raise ValueError(
            "Input file must be a DICOM file with .dcm or .dicom extension"
        )
    image = load_dicom(args.input)
    rescaled = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Check device availability
    if torch.xpu.is_available():
        device = "xpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    if args.kind == "otsu":
        bbox = get_ROIbox(rescaled)  # add channel dimension
    elif args.kind == "fcnn":
        # load detector model
        detector = Detector(modelpath=args.modelpath, device=device)
        bbox = detector.get_roi(rescaled[..., np.newaxis])

    print(f"Breast is located at: {bbox}")

    # obtain clahe-enhanced image
    clahed = clahefusion(rescaled, [1.0, 2.0])
    cropped = clahed[bbox[1] : bbox[3], bbox[0] : bbox[2]]

    # save to file
    outpath = path_join(args.output, f"cropped_{args.kind}.png")
    if not isdir(args.output):
        makedirs(args.output)
    cv2.imwrite(outpath, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    print(f"Cropped image saved to: {outpath}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Test detector models")
    parser.add_argument("input", type=str, help="Path to DICOM file")
    parser.add_argument(
        "--kind",
        "-k",
        choices=["otsu", "fcnn"],
        default="otsu",
        help="Kind of segmentation to apply",
    )
    parser.add_argument(
        "--modelpath",
        type=str,
        default="models/roi_detector.pth",
        help="Path to detector model",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="out/",
        help="Output directory to save results. it will be created if it does not exist.",
    )
    args = parser.parse_args()
    main(args)
