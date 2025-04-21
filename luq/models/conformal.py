import torch
import typing as T
from luq.llm import LLMSamples
from luq.models.base_uq_model import BaseUQModel
from luq.utils import SeqProbMode


class ConformalUQEstimator(BaseUQModel):
    def __init__(self, alpha: float = 0.1):
        """
        Initialize conformal predictor for uncertainty quantification.
        
        Args:
            alpha: Significance level (default 0.1 for 90% confidence)
        """
        super().__init__()
        self.alpha = alpha

    def compute_nonconformity_scores(
        self,
        samples: LLMSamples,
        seq_prob_mode: SeqProbMode = SeqProbMode.PROD,
    ) -> torch.Tensor:
        """
        Compute nonconformity scores for each sample based on sequence probabilities.
        """
        scores = []
        for sample in samples.samples:
            if sample.logprobs is not None:
                score = -self.compute_sequence_probability(sample.logprobs, seq_prob_mode)
                scores.append(score)
            else:
                raise ValueError("Logprobs required for conformal prediction")
        return torch.tensor(scores)

    def estimate_uncertainty(
        self,
        samples: LLMSamples,
        seq_prob_mode: SeqProbMode = SeqProbMode.PROD,
        **kwargs,
    ) -> float:
        """
        Estimate uncertainty using conformal prediction framework.
        
        Returns credibility score (1 - p-value) as uncertainty measure.
        Lower credibility indicates higher uncertainty.
        """
        scores = self.compute_nonconformity_scores(samples, seq_prob_mode)
        
        # Compute empirical p-value
        n = len(scores)
        calibration_scores = scores[:-1]  # Use n-1 samples for calibration
        test_score = scores[-1]  # Use last sample as test point
        
        p_value = torch.sum(calibration_scores >= test_score).item() / n
        
        # Return credibility score
        return 1.0 - p_value
    
