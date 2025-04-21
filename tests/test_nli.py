import torch

from luq.llm.nli import (
    hard_nli_clustering,
    NLIResult,
    NLIOutput,
    NLIWrapper,
    construct_nli_table,
)
from luq.llm.llm import LLMOutput, LLMSamples


def test_nli_output():
    probs = torch.tensor([0.1, 0.8, 0.1])
    output = NLIOutput(cls=NLIResult.ENTAILMENT, probs=probs)
    assert output.cls == NLIResult.ENTAILMENT
    assert torch.equal(output.probs, probs)


def test_nli_result():
    assert NLIResult.CONTRADICTION.value == "contradiction"
    assert NLIResult.ENTAILMENT.value == "entailment"
    assert NLIResult.NEUTRAL.value == "neutral"


def test_hard_nli_clustering():
    # Mock samples
    samples = LLMSamples(
        samples=[
            LLMOutput(answer="The sky is blue"),
            LLMOutput(answer="The sky is azure"),
            LLMOutput(answer="The grass is green"),
        ],
        answer=LLMOutput(answer="The sky is blue"),
        params={},
    )

    # Mock NLI model
    class MockNLI(NLIWrapper):
        def __call__(self, text1, text2, params=None):
            if "sky" in text1 and "sky" in text2:
                return NLIOutput(
                    cls=NLIResult.ENTAILMENT, probs=torch.tensor([0.1, 0.8, 0.1])
                )
            return NLIOutput(cls=NLIResult.NEUTRAL, probs=torch.tensor([0.1, 0.1, 0.8]))

    nli_model = MockNLI()
    nli_table = construct_nli_table(samples, nli_model)

    clusters = hard_nli_clustering(samples, nli_table)
    # First two samples should be in same cluster (both about sky)
    assert clusters[0] == clusters[1]
    # Third sample should be in different cluster
    assert clusters[2] != clusters[0]


def test_construct_nli_table():
    samples = LLMSamples(
        samples=[LLMOutput(answer="Sample 1"), LLMOutput(answer="Sample 2")],
        answer=LLMOutput(answer="Answer"),
        params={},
    )

    class MockNLI(NLIWrapper):
        def __call__(self, text1, text2, params=None):
            return NLIOutput(cls=NLIResult.NEUTRAL, probs=torch.tensor([0.1, 0.1, 0.8]))

    nli_model = MockNLI()
    nli_table = construct_nli_table(samples, nli_model)

    # Check table contains all sample pairs
    answer1, answer2 = samples.samples[0].answer, samples.samples[1].answer
    assert (answer1, answer2) in nli_table
    assert isinstance(nli_table[(answer1, answer2)], NLIOutput)
