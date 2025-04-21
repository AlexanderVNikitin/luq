import torch
import typing as T

from luq.llm import LLMWrapper, LLMOutput
from luq.models.base_uq_model import BaseUQModel
from luq.utils import entropy, SeqProbMode


class PredictiveEntropyEstimator(BaseUQModel):
    def generate_logits(self, prompt: str, num_samples: int = 10) -> T.List:
        """
        Generates multiple responses and extracts their logits.

        :param prompt: The input prompt for the LLM.
        :return: List of logit sequences.
        """
        logit_samples = []

        for _ in range(num_samples):
            if isinstance(self._llm, LLMWrapper):
                response = self._llm(prompt)
            else:
                raise ValueError(
                    f"Cannot compute logits LogitUncertaintyQuantification for {type(self._llm)}"
                )
            logit_samples.append(response.logits)

        return logit_samples

    def compute_entropy(self, sequence_probs):
        """
        Computes entropy over the sampled sequence probabilities.

        :param sequence_probs: List of sequence probabilities.
        :return: Entropy value.
        """
        if not isinstance(sequence_probs, torch.Tensor):
            sequence_probs = torch.tensor(sequence_probs)

        sequence_probs /= sum(
            sequence_probs
        )  # Normalize to form a probability distribution
        return entropy(sequence_probs)

    def estimate_uncertainty(
        self,
        samples: T.List[LLMOutput],
        seq_prob_mode: SeqProbMode = SeqProbMode.PROD,
        **kwargs,
    ) -> float:
        """
        Uncertainty is estimated by computing the entropy of probabilities obtained from sampled sequences.

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
        entropy_value = self.compute_entropy(sequence_probs)

        return entropy_value
