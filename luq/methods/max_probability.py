import typing as T

from luq.models import LLMOutput
from luq.methods.base_uq_model import BaseUQModel
from luq.utils import SeqProbMode


class MaxProbabilityEstimator(BaseUQModel):
    """Uncertainty estimator that uses the probability of the most likely sequence.

    This class estimates uncertainty by computing the probability of each sequence in a set of samples,
    and returning one minus the maximum probability, which serves as a measure of uncertainty.
    """
    def estimate_uncertainty(
        self,
        samples: T.List[LLMOutput],
        seq_prob_mode: SeqProbMode = SeqProbMode.PROD,
        **kwargs,
    ) -> float:
        """Estimate uncertainty from a list of LLM output samples.

        This method calculates the sequence probability for each sample using the specified
        sequence probability mode and returns an uncertainty score equal to `1 - max(sequence_probs)`.

        Args:
            samples (List[LLMOutput]): A list of language model outputs with associated log probabilities.
            seq_prob_mode (SeqProbMode, optional): Mode for aggregating token probabilities into
                sequence probabilities (e.g., product or average). Defaults to `SeqProbMode.PROD`.
            **kwargs: Additional keyword arguments (unused here but kept for compatibility).

        Returns:
            float: Uncertainty score, where higher values indicate more uncertainty.
        """
        assert all(s.logprobs is not None for s in samples.samples)

        logit_samples = [s.logprobs for s in samples.samples]
        sequence_probs = [
            self.compute_sequence_probability(logits, seq_prob_mode)
            for logits in logit_samples
        ]
        sequence_probs = self.normalize_sequence_probs(sequence_probs)
        return 1 - max(sequence_probs)
