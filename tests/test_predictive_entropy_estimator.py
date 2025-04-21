import numpy as np
import torch

from luq.models.predictive_entropy import PredictiveEntropyEstimator, SeqProbMode
from luq.llm import LLMOutput, LLMWrapper, generate_n_samples_and_answer


SAMPLE_LOGPROBS = [-1.0, -12.0, -7.2, -1.0, -12.0, -7.2, -1.0, -12.0, -7.2]
SAMPLE_PROBS = [np.exp(s) for s in SAMPLE_LOGPROBS]


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


def test_predictive_entropy_estimator():
    samples = gen_samples(24)  # TODO parameterize with num samples
    estimator = PredictiveEntropyEstimator()
    # Calculate entropy
    entropy = estimator.estimate_uncertainty(samples)

    # Expected entropy calculation
    probs = [SAMPLE_PROBS] * 24
    probs = np.prod(probs, axis=1)
    probs /= sum(probs)
    expected_entropy = get_entropy(probs)

    assert isinstance(entropy, float)
    assert np.isclose(entropy, expected_entropy, atol=1e-9)


def test_predictive_entropy_estimator_zero_entropy():
    confident_logprobs = [-1e15, -1e15, -1e15, -1e15, -1e15, -1e15]
    llm_mock = MockLLM(logprobs=confident_logprobs)
    confident_probs = [[0.0, 0.0, 0.0, 0.0] for _ in range(24)]
    confident_probs[0] = [1.0, 1.0, 1.0, 1.0]
    # Probability distribution with zero entropy (100% certain)
    samples = gen_samples(24, llm=llm_mock)  # TODO parameterize with num samples
    samples.samples[0].logprobs = torch.zeros(samples.samples[0].logprobs.shape)

    estimator = PredictiveEntropyEstimator()
    # Calculate entropy
    entropy = estimator.estimate_uncertainty(samples)

    probs = np.prod(confident_probs, axis=1)
    probs /= sum(probs)
    expected_entropy = get_entropy(probs)

    assert isinstance(entropy, float)
    assert np.isclose(entropy, expected_entropy, atol=1e-8)

    # now test avg mode
    entropy = estimator.estimate_uncertainty(samples, SeqProbMode.AVG)
    assert isinstance(entropy, float)
    assert np.isclose(entropy, expected_entropy, atol=1e-8)
