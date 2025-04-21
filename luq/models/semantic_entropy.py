import torch
import typing as T
from collections import Counter

from luq.llm import LLMSamples
from luq.llm.nli import (
    NLIWrapper,
    NLITable,
    construct_nli_table,
    hard_nli_clustering,
)
from luq.models.base_uq_model import BaseUQModel
from luq.utils import entropy, SeqProbMode


class SemanticEntropyEstimator(BaseUQModel):
    def __init__(self):
        super().__init__()

    def compute_entropy(
        self, cluster_assignments: T.List[int], sequence_probs: T.List[float] | None
    ) -> float:
        """
        Computes entropy over clusters using sequence probabilities or cluster sizes.
        """
        if sequence_probs is None:
            # Discrete Semantic Entropy
            cluster_counts = Counter(cluster_assignments)
            cluster_probs = torch.tensor(
                [
                    count / sum(cluster_counts.values())
                    for count in cluster_counts.values()
                ]
            )
        else:
            # Continuous Semantic Entropy with sequence probabilities
            cluster_probs = torch.zeros(max(cluster_assignments) + 1)
            for cluster_id, prob in zip(cluster_assignments, sequence_probs):
                cluster_probs[cluster_id] += prob
            # Normalize probabilities
            cluster_probs = cluster_probs / torch.sum(cluster_probs)

        return entropy(cluster_probs)

    def estimate_uncertainty(
        self,
        samples: LLMSamples,
        seq_prob_mode: SeqProbMode = SeqProbMode.PROD,
        nli_model: NLIWrapper | None = None,
        nli_table: NLITable | None = None,
        **kwargs,
    ) -> float:
        """
        Uncertainty is estimated by performing semantic clustering of LLM responses and calculating the entropy across the clusters.
        """

        # validation
        if nli_model is None and nli_table is None:
            raise ValueError("Either `nli_model` or `nli_table` should be provided")

        if nli_model is not None and nli_table is not None:
            raise ValueError(
                "Only one of `nli_model` and `nli_table` should be provided"
            )

        logit_samples = [s.logprobs for s in samples.samples]

        # Compute sequence probabilities
        sequence_probs = [
            self.compute_sequence_probability(logits, seq_prob_mode)
            for logits in logit_samples
        ]

        if nli_table is None:
            nli_table = construct_nli_table(samples, nli_model)

        # Cluster responses
        cluster_assignments = hard_nli_clustering(samples, nli_table)

        # Compute entropy over clusters
        return self.compute_entropy(cluster_assignments, sequence_probs)
