import os
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

from findclf import ImageIO
from findclf.Defaults import LABEL_NAMES
from findclf.ModelOps import Detector, load_classifier
from findclf.Transforms import get_transforms
from findclf.WindowOps import apply_overlap_kernel, batcher, create_kernel, get_windows


def _check_device() -> str:
    if torch.xpu.is_available():
        # Intel GPU
        return "xpu"
    elif torch.cuda.is_available():
        # Nvidia GPU
        return "cuda"
    elif torch.backends.mps.is_available():
        # Apple Silicon
        return "mps"
    else:
        return "cpu"


def main(args):
    # Check images and prepare output directory
    image_paths = ImageIO.get_image_paths(args.input)
    os.makedirs(args.outpath, exist_ok=True)

    # Check if model exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file {args.model} does not exist.")

    # Get device for computation
    device = _check_device()
    print(f"Using device: {device}")

    # Load detector
    # check if detector is available
    if args.roi:
        if os.path.exists(args.detector_model) and args.detector_type == "frcnn":
            detector = Detector(modelpath=args.detector_model, device=device)
            print("Using trained Faster R-CNN breast detector.")
        else:
            detector = None
            print("Using Otsu's thresholding for breast detection.")

    # Load classifier model
    classifier = load_classifier(
        modelpath=args.model, device=device, n_labels=len(LABEL_NAMES)
    )

    # Get transforms
    transform_ops = get_transforms(
        (args.window_size, args.window_size), expand_dims=False
    )
    kernel = create_kernel(args.window_size, args.stride)

    # Process each image
    for imfile in tqdm(image_paths, desc="Processing images", unit="img"):
        # make folder for image
        outfile_dir = os.path.join(args.outpath, os.path.basename(imfile))
        os.makedirs(outfile_dir, exist_ok=True)

        # load image
        if imfile.lower().endswith((".dcm", ".dicom")):
            image = ImageIO.load_dicom(imfile)
            image = (image * 255).astype(np.uint8)  # return to 0-255

        else:
            image = ImageIO.load_image(imfile)
            # we assume that if a non-dicom image is loaded is gray-scale and in 0-255 range

        if args.roi:
            # get breast ROI
            if detector is not None:
                bbox = detector.get_roi(image[..., np.newaxis])
            else:
                bbox = ImageIO.get_ROIbox(image)

        # Apply CLAHE fusion
        image = ImageIO.clahefusion(image, thresholds=[1.0, 2.0])
        # crop to ROI if applicable
        if args.roi:
            image = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            # save the original cropped image for reference
            ImageIO.save_image(
                image,
                os.path.join(outfile_dir, f"cropped_{os.path.basename(imfile)}.png"),
            )

        # divide the image into windows
        windows, (dx, dy) = get_windows(image, args.window_size, args.stride)
        # Filter empty windows
        not_empty = windows.sum(axis=(1, 2, 3)) != 0
        not_empty_indices = np.argwhere(not_empty).flatten()

        # create an array to hold all predictions, we'll only fill those with non-empty windows, as given by not_empty_indices
        all_predictions = np.zeros((len(windows), len(LABEL_NAMES)))

        window_predictions = []  # store predictions
        total_windows_per_batch = len(not_empty_indices) // args.batch_size

        for window in tqdm(
            batcher(windows[not_empty_indices], args.batch_size),
            total=total_windows_per_batch,
            desc="Infering windows",
            unit="window",
            unit_scale=args.batch_size,
            leave=False,
        ):
            # convert to tensor
            with torch.no_grad():
                window_tensor = torch.from_numpy(window)
                window_tensor = transform_ops(window_tensor)
                window_tensor = window_tensor.to(device)
                # infer
                outputs = classifier(window_tensor)  # forward pass
                prediction = torch.sigmoid(
                    outputs["Classifier"]
                )  # apply sigmoid to get probabilities
                window_predictions.append(prediction.detach().cpu().numpy())

        # after all batches are processed
        # check if theres NaNs in predictions
        window_predictions = np.nan_to_num(np.vstack(window_predictions))
        # copy resulto to the all_predictions array
        all_predictions[not_empty_indices] = (
            window_predictions.copy()
        )  # copy for safety
        heatmaps = all_predictions.reshape(len(dx), len(dy), len(LABEL_NAMES))

        # convolve with kernel to smooth heatmaps
        for idx_label, label in enumerate(LABEL_NAMES):
            heatmap = heatmaps[:, :, idx_label]
            heatmap = apply_overlap_kernel(
                heatmap, kernel, image.shape[:2][::-1], args.window_size
            )

            # if mask option is set, output binary mask
            if args.mask:
                binary_mask = ((heatmap >= args.threshold) * 255).astype(np.uint8)
                ImageIO.save_image(
                    binary_mask,
                    os.path.join(
                        outfile_dir, f"{os.path.basename(imfile)}_{label}_mask.png"
                    ),
                )
            else:
                # else, output heatmap overlayed on original image
                overlay = ImageIO.mix_heatmap(
                    image, (heatmap * 255).astype(np.uint8), alpha=0.5
                )
                ImageIO.save_image(
                    overlay,
                    os.path.join(
                        outfile_dir, f"{os.path.basename(imfile)}_{label}_heatmap.png"
                    ),
                )

    print("Processing completed.")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Inspect a breast image for locating pathological findings using a sliding window classifier."
    )
    parser.add_argument(
        "input",
        nargs="+",
        type=str,
        help="Path to image file or directory containing images to process.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/EfficientNetV2_final.pth",
        help="Path to the trained classifier model.",
    )
    parser.add_argument(
        "--detector-type",
        "-d",
        choices=["otsu", "frcnn"],
        default="otsu",
        help="Type of breast detector to use: 'otsu' for Otsu's thresholding, 'frcnn' for a trained Faster R-CNN model. Default is Otsu.",
    )
    parser.add_argument(
        "--detector-model",
        type=str,
        default="models/roi_detector.pth",
        help="Path to the trained breast detector model (used if --detector-type is 'frcnn').",
    )
    parser.add_argument(
        "--window-size",
        "-w",
        type=int,
        default=256,
        help="Window size (in pixels) for the sliding window. Default is 256.",
    )
    parser.add_argument(
        "--stride",
        "-s",
        type=int,
        default=32,
        help="Stride (in pixels) for the sliding window. Default is 32.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Batch size for processing windows. Default is 32.",
    )
    parser.add_argument(
        "--outpath",
        "-o",
        type=str,
        default="out/",
        help="Output directory to save the results. Default is 'out/'.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="Threshold for classifying a window as positive. Default is 0.5.",
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="If set, output a mask image indicating detected regions. By default will output heatmaps overlaid on the original image.",
    )
    parser.add_argument(
        "--roi",
        action="store_true",
        help="If set, use breast ROI to limit sliding window area (requires detector).",
    )
    args = parser.parse_args()
    main(args)
