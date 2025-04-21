import numpy as np
import torch

from luq.models.semantic_entropy import SemanticEntropyEstimator
from luq.llm import LLMOutput, LLMWrapper, generate_n_samples_and_answer, LLMSamples
from luq.llm.nli import NLIWrapper, NLIResult, NLIOutput, construct_nli_table
from luq.utils import SeqProbMode


SAMPLE_LOGPROBS = [-1.0, -12.0, -7.2, -1.0, -12.0, -7.2, -1.0, -12.0, -7.2]
SAMPLE_PROBS = [np.exp(s) for s in SAMPLE_LOGPROBS]


def get_samples_50_50():
    answers = [LLMOutput("sky", torch.tensor(SAMPLE_LOGPROBS)) for _ in range(5)] + [
        LLMOutput("earth", torch.tensor(SAMPLE_LOGPROBS)) for _ in range(5)
    ]
    return LLMSamples(answers, answer=answers[0], params={})


def get_entropy(probs):
    return -np.sum(probs * np.log((probs + 1e-8)))


class MockLLM(LLMWrapper):
    def __init__(self, logprobs=SAMPLE_LOGPROBS):
        self.logprobs = logprobs

    def __call__(
        self, prompt: str, top_p: float, top_k: float, temperature: float
    ) -> LLMOutput:
        return LLMOutput(
            f"It is the final answer to prompt <<{prompt}>>",
            logprobs=torch.tensor(self.logprobs),
        )


# Mock NLI model
class MockNLI(NLIWrapper):
    def __call__(self, text1, text2, params=None):
        if "sky" in text1 and "sky" in text2 or "earth" in text1 and "earth" in text2:
            return NLIOutput(
                cls=NLIResult.ENTAILMENT, probs=torch.tensor([0.1, 0.8, 0.1])
            )
        return NLIOutput(cls=NLIResult.NEUTRAL, probs=torch.tensor([0.1, 0.1, 0.8]))


def gen_samples(n_samples=10, llm=None):
    llm = llm or MockLLM()

    params = {
        "prompt": "test prompt",
        "temp_gen": 0.7,
        "temp_answer": 0.1,
        "top_p_gen": 0.9,
        "top_k_gen": 40,
        "top_p_ans": 0.9,
        "top_k_ans": 40,
        "n_samples": n_samples,
    }

    samples = generate_n_samples_and_answer(llm=llm, **params)
    return samples


def test_semantic_entropy_estimator():
    samples = gen_samples(24)  # TODO parameterize with num samples
    estimator = SemanticEntropyEstimator()
    nli_model = MockNLI()
    nli_table = construct_nli_table(samples, nli_model)
    # Calculate entropy
    entropy = estimator.estimate_uncertainty(samples, nli_table=nli_table)

    # Expected entropy calculation
    probs = [SAMPLE_PROBS] * 24
    probs = np.prod(probs, axis=1)
    probs /= sum(probs)
    expected_entropy = get_entropy(probs)

    assert isinstance(entropy, float)
    assert np.isclose(entropy, expected_entropy, atol=1e-9)


def test_semantic_entropy_estimator_zero_entropy():
    confident_logprobs = [-1e15, -1e15, -1e15, -1e15, -1e15, -1e15]
    llm_mock = MockLLM(logprobs=confident_logprobs)
    confident_probs = [[0.0, 0.0, 0.0, 0.0] for _ in range(24)]
    confident_probs[0] = [1.0, 1.0, 1.0, 1.0]
    # Probability distribution with zero entropy (100% certain)
    samples = gen_samples(24, llm=llm_mock)  # TODO parameterize with num samples
    samples.samples[0].logprobs = torch.zeros(samples.samples[0].logprobs.shape)

    nli_model = MockNLI()
    nli_table = construct_nli_table(samples, nli_model)

    estimator = SemanticEntropyEstimator()
    # Calculate entropy
    entropy = estimator.estimate_uncertainty(samples, nli_table=nli_table)

    probs = np.prod(confident_probs, axis=1)
    probs /= sum(probs)
    expected_entropy = get_entropy(probs)

    assert isinstance(entropy, float)
    assert np.isclose(entropy, expected_entropy, atol=1e-8)

    # now test avg mode
    entropy = estimator.estimate_uncertainty(
        samples, SeqProbMode.AVG, nli_table=nli_table
    )
    assert isinstance(entropy, float)
    assert np.isclose(entropy, expected_entropy, atol=1e-8)


def test_semantic_entropy_estimator_samples_50_50():
    samples = get_samples_50_50()

    nli_model = MockNLI()
    nli_table = construct_nli_table(samples, nli_model)

    estimator = SemanticEntropyEstimator()
    # Calculate entropy
    entropy = estimator.estimate_uncertainty(samples, nli_table=nli_table)

    assert isinstance(entropy, float)
    assert np.isclose(
        entropy, 2 * (-0.5 * torch.log(torch.tensor(0.5))).item(), atol=1e-8
    )
