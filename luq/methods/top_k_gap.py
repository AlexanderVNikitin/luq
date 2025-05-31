import typing as T

from luq.models import LLMOutput
from luq.methods.base_uq_model import BaseUQModel
from luq.utils import SeqProbMode


class TopKGapEstimator(BaseUQModel):
    def estimate_uncertainty(
        self,
        samples: T.List[LLMOutput],
        seq_prob_mode: SeqProbMode = SeqProbMode.PROD,
        k: int = 2,
        **kwargs,
    ) -> float:
        """Estimates uncertainty using the gap between the top-k sequence probabilities.

        The method computes sequence-level probabilities from the sampled responses,
        identifies the `k` highest probabilities, and returns a normalized uncertainty
        score as `1 - (gap between top-1 and top-k probabilities)`.

        A smaller gap between top-k and top-1 implies higher uncertainty (less confident top choice),
        while a large gap suggests stronger model confidence.

        Args:
            samples (List[LLMOutput]): A list of LLM outputs containing log-probabilities.
            seq_prob_mode (SeqProbMode, optional): Method for combining token log-probabilities into
                a single sequence probability. Defaults to `SeqProbMode.PROD`.
            k (int, optional): The number of top probabilities to compare. Must be >= 2. Defaults to 2.
            **kwargs: Additional arguments for extensibility (unused).

        Returns:
            float: A normalized uncertainty score based on the gap between top-1 and top-k probabilities.

        Raises:
            ValueError: If `k` is less than 2.
            AssertionError: If any sample does not contain log-probabilities.
        """
        if k < 2:
            raise ValueError("k should >= 2")
        assert all(s.logprobs is not None for s in samples.samples)

        logit_samples = [s.logprobs for s in samples.samples]
        sequence_probs = [
            self.compute_sequence_probability(logits, seq_prob_mode)
            for logits in logit_samples
        ]
        sorted_seq_probs = sorted(sequence_probs)
        gap = sorted_seq_probs[-1] - sorted_seq_probs[-k]
        return 1 - gap
