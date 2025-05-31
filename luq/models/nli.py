import torch
from dataclasses import dataclass
from enum import Enum
import typing as T

from luq.models.llm import LLMSamples


class NLIResult(Enum):
    """
    Enum representing the possible outcomes of a Natural Language Inference (NLI) model.

    Attributes:
        CONTRADICTION: The two inputs contradict each other.
        ENTAILMENT: One input entails the other.
        NEUTRAL: The relationship between the inputs is neutral.
    """
    CONTRADICTION = "contradiction"
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"


@dataclass
class NLIOutput:
    """
    Represents the output of an NLI model.

    Attributes:
        cls (NLIResult): The predicted NLI class.
        probs (torch.Tensor): A tensor of probabilities for each class.
    """
    cls: NLIResult
    probs: torch.Tensor


class NLIWrapper:
    """
    Abstract wrapper class for Natural Language Inference (NLI) models.
    """
    def __call__(*args, **kwargs) -> T.List[NLIOutput]:
        """
        Runs the NLI model on input arguments.

        Returns:
            List[NLIOutput]: A list of NLI model outputs.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError("NLI model should implement `__call__` method.")


NLITable = T.Dict[T.Tuple[str, str], NLIOutput]


def construct_nli_table(samples: LLMSamples, nli_model: NLIWrapper) -> NLITable:
    """
    Constructs a table of NLI results for all pairs of generated samples.

    Args:
        samples (LLMSamples): The generated language model outputs.
        nli_model (NLIWrapper): An NLI model wrapper used to evaluate relationships between outputs.

    Returns:
        NLITable: A dictionary mapping (answer1, answer2) pairs to NLIOutput results.
    """
    result = {}
    for i, s1 in enumerate(samples.samples):
        for s2 in samples.samples:
            answer1, answer2 = s1.answer, s2.answer
            if (answer1, answer2) in result:
                continue
            nli_output: NLIOutput = nli_model(answer1, answer2, params=samples.params)
            result[(answer1, answer2)] = nli_output
    return result


def hard_nli_clustering(samples: LLMSamples, nli_table: NLITable) -> T.List[int]:
    """
    Performs hard clustering of samples based on mutual entailment using NLI results.

    Args:
        samples (LLMSamples): The list of LLM-generated samples.
        nli_table (NLITable): A dictionary of NLI outputs between sample pairs.

    Returns:
        List[int]: A list of cluster assignments (by index) for each sample.
    """
    clusters = [None] * len(samples.samples)
    last_cluster = 0
    for i, s1 in enumerate(samples.samples):
        if clusters[i] is None:
            clusters[i] = last_cluster
            last_cluster += 1
        for j, s2 in enumerate(samples.samples[i + 1 :], i + 1):
            if clusters[j] is not None:
                continue
            if (
                nli_table[(s1.answer, s2.answer)].cls == NLIResult.ENTAILMENT
                and nli_table[(s2.answer, s1.answer)].cls == NLIResult.ENTAILMENT
            ):
                clusters[j] = clusters[i]
    return clusters
