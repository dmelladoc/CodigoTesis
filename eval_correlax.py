import os
from argparse import ArgumentParser

import numpy as np
import torch
from cv2 import COLORMAP_JET, COLORMAP_TWILIGHT
from tqdm import tqdm

from correlax.Algorithm import CorRELAX
from findclf import ImageIO
from findclf.Defaults import LABEL_NAMES
from findclf.ModelOps import Detector, load_classifier
from findclf.Transforms import get_transforms
from findclf.WindowOps import (
    apply_overlap_kernel,
    create_kernel,
    get_windows,
    resize_map_to_image,
)


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
        (args.window_size, args.window_size), expand_dims=True
    )
    kernel = create_kernel(args.window_size, args.stride)

    # initialize CorRELAX explainer
    crxr = CorRELAX(
        model=classifier,
        mask_batchsize=args.batch_size,
        mask_iter=args.iter_masks,
        window_size=(args.window_size, args.window_size),
        device=device,
        cell_size=(8, 8),
        p=0.5,
    )

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

        # NOTE: By design, we inspect each window individually for explanation. If more memory is available, expand args.batch_size. This will improve precision of CorRELAX explanations. If you need more speed, reduce args.batch_size and/or args.iter_masks. At cost of precision.
        window_outputs = {
            "Predictions": [],
            "CorrDist": [],
            "CorrPred": [],
        }

        for window in tqdm(windows, desc="Explaining windows", unit="win", leave=False):
            with torch.no_grad():
                # Convert window to tensor and apply transformations
                window_tensor = torch.from_numpy(window)
                window_tensor = transform_ops(window_tensor)
                window_tensor = window_tensor.to(device)

                # Obtain CorRELAX explanations
                outputs = crxr(window_tensor)
                # Append each one to their corresponding list
                window_outputs["Predictions"].append(
                    outputs["ImagePrediction"].detach().tolist()
                )
                window_outputs["CorrDist"].append(outputs["CorrDist"].item())
                window_outputs["CorrPred"].append(outputs["CorrPred"].detach().tolist())

        # Filter nans and repair shape
        mat_outputs = {}
        for name, values in window_outputs.items():
            if name == "CorrDist":
                shape = (len(dx), len(dy))
            else:
                shape = (len(dx), len(dy), len(LABEL_NAMES))
            mat_outputs[name] = np.nan_to_num(np.array(values)).reshape(*shape)

        # Plot correlation and prediction maps
        for idx_label, label in enumerate(LABEL_NAMES):
            # Prediction map
            pred_map = mat_outputs["Predictions"][:, :, idx_label]
            pred_map = apply_overlap_kernel(
                pred_map, kernel, image.shape[:2][::-1], args.window_size
            )
            pred_overlay = ImageIO.mix_heatmap(
                image, (pred_map * 255).astype(np.uint8), alpha=0.5, cmap=COLORMAP_JET
            )
            ImageIO.save_image(
                pred_overlay,
                os.path.join(
                    outfile_dir,
                    f"predmap_{label}_{os.path.basename(imfile)}.png",
                ),
            )

            # Correlation map
            # TODO: verify normalization, looks weird at output. Maybe normalize between max of absolute values?
            corr_map = mat_outputs["CorrPred"][:, :, idx_label]
            # Normalize from -1 to 1 into 0-255
            corr_map = ((corr_map + 1) / 2 * 255).astype(np.uint8)
            # continue processing
            corr_map = apply_overlap_kernel(
                corr_map, kernel, image.shape[:2][::-1], args.window_size
            )
            corr_overlay = ImageIO.mix_heatmap(
                image,
                (corr_map * 255).astype(np.uint8),
                alpha=0.5,
                cmap=COLORMAP_TWILIGHT,
            )
            ImageIO.save_image(
                corr_overlay,
                os.path.join(
                    outfile_dir,
                    f"corrmap_{label}_{os.path.basename(imfile)}.png",
                ),
            )

        # And plot distance correlation map
        corr_dist_map = mat_outputs["CorrDist"]
        corr_dist_map = resize_map_to_image(
            corr_dist_map, image.shape[:2][::-1], args.window_size
        )
        corrdist_overlay = ImageIO.mix_heatmap(
            image,
            (corr_dist_map * 255).astype(np.uint8),
            alpha=0.7,
            cmap=COLORMAP_PLASMA,  # changed to plasma for better visibility
        )
        ImageIO.save_image(
            corrdist_overlay,
            os.path.join(
                outfile_dir,
                f"corrdistmap_{os.path.basename(imfile)}.png",
            ),
        )

    print("Processing completed.")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Obtain CorRELAX explanations for our sliding window classifier."
    )
    parser.add_argument(
        "input",
        nargs="+",
        type=str,
        help="Path to input image or directory of images.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/EfficientNetV2_final.pth",
        help="Path to the trained classifier model.",
    )
    parser.add_argument(
        "--roi",
        action="store_true",
        help="If set, use breast ROI to limit sliding window area (requires detector).",
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
        "--iter-masks",
        type=int,
        default=40,
        help="Number of iterations for mask generation in CorRELAX. Default is 40.",
    )
    # TODO: add option argument for output the maps without overlay. Useful for testing issues with normalization of correlation maps
    args = parser.parse_args()
    main(args)
