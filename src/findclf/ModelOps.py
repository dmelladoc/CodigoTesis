from os.path import isfile

import torch
from torch import nn
from torchvision.models import efficientnet_v2_s
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import v2 as transforms


class Detector:
    """
    Creates an object detector using a pre-trained model.

    Args:
        modelpath (str): Path to the pre-trained model file.
        device (str): Device to load the model on.

    Returns:
        Detector: An instance of the Detector class.

    Raises:
        FileNotFoundError: If the model file is not found at the specified path.
    """

    def __init__(self, modelpath: str, device: str = "cpu"):
        if not isfile(modelpath):
            raise FileNotFoundError(f"Model file not found at {modelpath}")

        # load model
        # WARNING: this method is unsafe. We should adapt this to load a state_dict instead.
        self.detector = torch.load(modelpath, weights_only=False, map_location=device)
        self.detector.eval()

        # Transformations for input images
        self.transforms = transforms.Compose(
            [
                transforms.ToPureTensor(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )
        self.device = device

    def get_roi(self, image) -> torch.Tensor:
        with torch.no_grad():
            image_tensor = torch.tensor(image).permute(2, 0, 1)
            image_tensor = self.transforms(image_tensor).to(self.device)
            prediction = self.detector([image_tensor])
            boxes = prediction[0]["boxes"].cpu().numpy().astype(int)
            return boxes[0]  # return the highest confidence box


def _create_EfficientNetV2(n_labels: int) -> nn.Module:
    model = efficientnet_v2_s(weights="IMAGENET1K_V1")

    # modify last layer to adapt to our model
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=1280, out_features=512, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=512, out_features=n_labels, bias=False),
    )

    # Use feature extractor to get feature layer
    return_nodes = {"avgpool": "Features", "classifier.3": "Classifier"}
    featmodel = create_feature_extractor(model, return_nodes=return_nodes)
    return featmodel


def load_classifier(
    modelpath: str, device: str = "cpu", n_labels: int = 8
) -> nn.Module:
    """
    Loads a pre-trained EfficientNetV2 classifier model, modified for a specific number of labels
    """
    if not isfile(modelpath):
        raise FileNotFoundError(f"Model file not found at {modelpath}")

    # create model architecture
    model = _create_EfficientNetV2(n_labels=n_labels)

    # load weights
    state_dict = torch.load(modelpath, map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)
    model.eval()

    return model
