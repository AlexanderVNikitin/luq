import torch
from typing import List, Union
from enum import Enum


dtype_map = {
    "float32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float64": torch.float64,
    "double": torch.float64,
    "int32": torch.int32,
    "int": torch.int32,
    "int64": torch.int64,
    "long": torch.int64,
}


class SeqProbMode(Enum):
    """
    Enumeration for modes of combining token probabilities in a sequence.

    Attributes:
        PROD: Use the product of probabilities.
        AVG: Use the average of probabilities.
    """
    PROD = "prod"
    AVG = "avg"


def entropy(probabilities: Union[List[float], torch.Tensor]) -> torch.Tensor:
    """
    Computes the entropy of a probability distribution.

    Args:
        probabilities (Union[List[float], torch.Tensor]): A list or tensor of probabilities.
            The probabilities should sum to 1 and represent a valid distribution.

    Returns:
        torch.Tensor: The computed entropy as a scalar tensor value.

    Notes:
        Adds a small epsilon (1e-9) to probabilities to avoid log(0).
    """
    probabilities = (
        torch.tensor(probabilities, dtype=torch.float32)
        if isinstance(probabilities, list)
        else probabilities
    )
    entropy_value = -torch.sum(probabilities * torch.log(probabilities + 1e-9))
    return entropy_value.item()
