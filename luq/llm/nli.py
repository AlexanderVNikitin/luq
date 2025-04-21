import torch
from dataclasses import dataclass
from enum import Enum
import typing as T

from luq.llm.llm import LLMSamples


class NLIResult(Enum):
    CONTRADICTION = "contradiction"
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"


@dataclass
class NLIOutput:
    cls: NLIResult
    probs: torch.Tensor


class NLIWrapper:
    def __call__(*args, **kwargs) -> T.List[NLIOutput]:
        raise NotImplementedError("NLI model should implement `__call__` method.")


NLITable = T.Dict[T.Tuple[str, str], NLIOutput]


def construct_nli_table(samples: LLMSamples, nli_model: NLIWrapper) -> NLITable:
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
