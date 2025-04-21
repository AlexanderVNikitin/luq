import torch
from typing import List, Union
from enum import Enum


class SeqProbMode(Enum):
    PROD = "prod"
    AVG = "avg"


def entropy(probabilities: Union[List[float], torch.Tensor]) -> torch.Tensor:
    """Compute entropy for a sequence of probabilities."""
    probabilities = (
        torch.tensor(probabilities, dtype=torch.float32)
        if isinstance(probabilities, list)
        else probabilities
    )
    entropy_value = -torch.sum(probabilities * torch.log(probabilities + 1e-9))
    return entropy_value.item()
