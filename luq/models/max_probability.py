import typing as T

from luq.llm import LLMOutput
from luq.models.base_uq_model import BaseUQModel
from luq.utils import SeqProbMode


class MaxProbabilityEstimator(BaseUQModel):
    def estimate_uncertainty(
        self,
        samples: T.List[LLMOutput],
        seq_prob_mode: SeqProbMode = SeqProbMode.PROD,
        **kwargs,
    ) -> float:
        """
        Estimates uncertainty as one minus the probability of the most likely sequence in the list of samples.

        :param prompt: The input prompt for LLM.
        :param seq_prob_mode: Describes how token probabilities are translated into sequence probabilities
        :return: entropy score
        """
        assert all(s.logprobs is not None for s in samples.samples)

        logit_samples = [s.logprobs for s in samples.samples]
        sequence_probs = [
            self.compute_sequence_probability(logits, seq_prob_mode)
            for logits in logit_samples
        ]
        sequence_probs = self.normalize_sequence_probs(sequence_probs)
        return 1 - max(sequence_probs)
