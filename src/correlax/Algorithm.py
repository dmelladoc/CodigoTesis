from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class CorRELAX(nn.Module):
    """
    CorRELAX
    Implementation of the CorRELAX algorithm for model interpretability.

    Args:
        model (nn.Module): The neural network model to be interpreted. It should return a dictionary with keys 'Classifier' and 'Features' (See Notes for details).
        mask_batchsize (int): Number of masks to generate per iteration.
        mask_iter (int): Number of iterations for mask generation.
        distance_fn (nn.Module): Distance function to compute similarity between feature vectors.
        window_size (Tuple[int, int]): Size of the input image window.
        device (str): Device to run the computations on ('cpu' or 'cuda').
        cell_size (Tuple[int, int]): Size of the cells for mask generation.
        p (float): Probability of masking a cell.

    Returns:
        dict: A dictionary containing:
            - 'ImagePrediction': The model's prediction for the original image.
            - 'CorrDist': Correlation coefficient between distance vectors and prediction vectors.
            - 'CorrPred': Correlation coefficients between distance vectors and predictions from masked images.

    Notes:
        The model should output a dictionary with at least two keys:
        'Classifier' for the final prediction and 'Features' for the internal feature vectors. This is built using `torchvision.models.feature_extraction.create_feature_extractor`, and retunring nodes as follows using torchvision implementation of EfficientNetV2:
            ```python
            return_nodes = {'avgpool': 'Features', 'classifier.3': 'Classifier'}
            featmodel = create_feature_extractor(efficientnet_v2_s(pretrained=True), return_nodes=return_nodes)
            ```
    """

    def __init__(
        self,
        model: nn.Module,
        mask_batchsize: int = 32,
        mask_iter: int = 40,
        distance_fn: nn.Module = nn.CosineSimilarity(dim=1),
        window_size: Tuple[int, int] = (256, 256),
        device: str = "cpu",
        cell_size: Tuple[int, int] = (8, 8),
        p: float = 0.5,
    ):
        super(CorRELAX, self).__init__()
        self.model = model
        self.mask_batchsize = mask_batchsize
        self.mask_iter = mask_iter
        self.distance_fn = distance_fn
        self.window_size = window_size
        self.device = device
        self.cell_size = cell_size
        self.p = p

    def _make_masks(self):
        for _ in range(self.mask_iter):
            mask = (
                torch.rand(self.mask_batchsize, 1, *self.cell_size, device=self.device)
                > self.p
            ).float()
            interp = F.interpolate(
                mask, size=self.window_size, mode="bilinear", align_corners=False
            )
            yield interp

    def forward(self, window: torch.Tensor):
        result = {name: [] for name in ["distVect", "predVect", "prediction"]}

        # get the image prediction and internal feature vectors
        output = self.model(window)
        image_pred = torch.sigmoid(output["Classifier"])
        image_vect = output["Features"]

        # Then, we mask the image, get their predictions and feature vectors
        for masks in self._make_masks():
            masked_window = window * masks
            masked_output = self.model(masked_window)
            masked_pred = torch.sigmoid(masked_output["Classifier"])
            masked_vect = masked_output["Features"]

            # Compute similarity distance between the original and masked feature vectors
            dist_vectors = self.distance_fn(image_vect, masked_vect).squeeze()
            result["distVect"].append(dist_vectors)

            # Compute the similarity between the original and masked predictions
            pred_vectors = self.distance_fn(image_pred, masked_pred).squeeze()
            result["predVect"].append(pred_vectors)

            # Accumulate masked predictions
            result["prediction"].append(masked_pred)

        # Concatenate results
        result["distVect"] = torch.cat(result["distVect"], dim=0)
        result["predVect"] = torch.cat(result["predVect"], dim=0)
        result["prediction"] = torch.cat(result["prediction"], dim=0)

        # Obtain correlation between distance vectors and prediction vectors
        corr_dist_mat = torch.corrcoef(
            torch.stack([result["distVect"], result["predVect"]], dim=0)
        )
        corr_dist = corr_dist_mat[0, 1]  # Correlation coefficient

        # The same but between distance vectors and predictions from masked images
        corr_pred_mat = torch.corrcoef(
            torch.cat([result["distVect"].unsqueeze(1), result["prediction"]], dim=1).T
        )
        corr_pred = corr_pred_mat[0, 1:]  # Correlation coefficients for each class

        return {
            "ImagePrediction": image_pred,
            "CorrDist": corr_dist,
            "CorrPred": corr_pred,
        }
