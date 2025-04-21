import torch
from luq.utils import SeqProbMode

from typing import List


class BaseUQModel:
    def compute_sequence_probability(
        self, logprobs: torch.Tensor, seq_prob_mode: SeqProbMode = SeqProbMode.PROD
    ) -> float:
        """
        Computes the probability of a response sequence.
        """
        token_probs = torch.exp(logprobs)  # Convert logits to probabilities
        if seq_prob_mode == SeqProbMode.PROD:
            return torch.prod(token_probs).item()
        elif seq_prob_mode == SeqProbMode.AVG:
            return torch.mean(token_probs).item()
        else:
            raise ValueError(f"Unknown seq_prob_mode: {seq_prob_mode}")

    def normalize_sequence_probs(
        self, probs: List[float], tolerance: float = 1e-9
    ) -> float:
        z = sum(probs)
        if abs(z) < tolerance:
            return [1.0 / len(probs)] * len(probs)
        return [p / z for p in probs]

    def estimate_uncertainty(self, prompt: str, *args, **kwargs) -> float:
        raise NotImplementedError("method get_uncertainty is not implemented")
