import typing as T

from luq.llm import LLMOutput
from luq.models.base_uq_model import BaseUQModel
from luq.utils import SeqProbMode


class TopKGapEstimator(BaseUQModel):
    def estimate_uncertainty(
        self,
        samples: T.List[LLMOutput],
        seq_prob_mode: SeqProbMode = SeqProbMode.PROD,
        k: int = 2,
        **kwargs,
    ) -> float:
        """
        Estimates uncertainty by computing entropy over sampled sequence probabilities.

        :param prompt: The input prompt for LLM.
        :param seq_prob_mode: Describes how token probabilities are translated into sequence probabilities
        :return: entropy score
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
