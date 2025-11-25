from argparse import ArgumentParser
from os import makedirs
from os.path import isdir, isfile, splitext
from os.path import join as path_join

import cv2
import numpy as np

from correlax.ImageIO import clahefusion, load_dicom, load_image


def main(args):
    # check if file exists
    if not isfile(args.input):
        raise FileNotFoundError(f"File {args.input} not found")

    # open DICOM File and extract the image
    if splitext(args.input)[1].lower() in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        image = load_image(args.input)
    elif splitext(args.input)[1].lower() in [".dcm", ".dicom"]:
        image = load_dicom(args.input)
    else:
        raise ValueError("Unsupported file format")

    # Reescale to 0-255 and convert to uint8 previous to CLAHE
    reescaled = cv2.normalize(
        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

    clahed = clahefusion(reescaled, [1.0, 2.0])
    outpath = path_join(args.output, "out.png")
    # create output directory if it does not exist
    if not isdir(args.output):
        makedirs(args.output, exist_ok=True)

    cv2.imwrite(outpath, cv2.cvtColor(clahed, cv2.COLOR_RGB2BGR))
    print(f"Processed image saved to {outpath}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Test image loading and processing")
    parser.add_argument("input", type=str, help="Path to input image file")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="out/",
        help="Path to output directory, will be created if it does not exist",
    )
    args = parser.parse_args()
    main(args)
