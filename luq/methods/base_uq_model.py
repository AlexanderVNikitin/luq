import torch
from luq.utils import SeqProbMode

from typing import List


class BaseUQModel:
    def compute_sequence_probability(
        self, logprobs: torch.Tensor, seq_prob_mode: SeqProbMode = SeqProbMode.PROD
    ) -> float:
        """
        Computes the probability of a response sequence from log-probabilities.

        Args:
            logprobs (torch.Tensor): A tensor containing log-probabilities of each token in the sequence.
            seq_prob_mode (SeqProbMode, optional): The method to compute the sequence probability.
                Options are SeqProbMode.PROD for product and SeqProbMode.AVG for average.
                Defaults to SeqProbMode.PROD.

        Returns:
            float: The computed sequence probability.

        Raises:
            ValueError: If an unknown `seq_prob_mode` is provided.
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
    ) -> List[float]:
        """
        Normalizes a list of sequence probabilities so they sum to 1.

        Args:
            probs (List[float]): A list of raw sequence probabilities.
            tolerance (float, optional): A small threshold below which the sum is considered zero
                to avoid division by zero. Defaults to 1e-9.

        Returns:
            List[float]: A list of normalized probabilities summing to 1.
        """
        z = sum(probs)
        if abs(z) < tolerance:
            return [1.0 / len(probs)] * len(probs)
        return [p / z for p in probs]

    def estimate_uncertainty(self, prompt: str, *args, **kwargs) -> float:
        """
        Estimates the uncertainty for a given prompt.

        Args:
            prompt (str): The input prompt to estimate uncertainty for.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            float: The estimated uncertainty value.

        Raises:
            NotImplementedError: This method must be implemented in a subclass.
        """
        raise NotImplementedError("method get_uncertainty is not implemented")
