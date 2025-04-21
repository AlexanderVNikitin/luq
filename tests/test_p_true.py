import pytest
import torch
from luq.models.p_true import PTrueEstimator
from luq.llm import LLMSamples, LLMOutput


@pytest.fixture
def p_true_estimator():
    class MockLLM:
        probs = torch.tensor([-2.0, -2.0])

        def __call__(self, prompt, temperature=None):
            return LLMOutput(answer="A", logprobs=torch.tensor(self.probs))

    estimator = PTrueEstimator(llm=MockLLM())
    return estimator


@pytest.fixture
def mock_samples():
    samples = [
        LLMOutput(answer="Sample 1", logprobs=torch.tensor([0.1, 0.2])),
        LLMOutput(answer="Sample 2", logprobs=torch.tensor([0.2, 0.3])),
        LLMOutput(answer="Sample 3", logprobs=torch.tensor([0.3, 0.4])),
    ]
    return LLMSamples(samples=samples, answer=samples[0], params={})


def test_construct_p_true_prompt(p_true_estimator):
    prompt = p_true_estimator.construct_p_true_prompt(
        question="Test question?",
        most_probable_answer="Answer 1",
        brainstormed_answers=["Answer 2", "Answer 3"],
    )
    assert "Test question?" in prompt
    assert "Answer 1" in prompt
    assert "Answer 2" in prompt
    assert "Answer 3" in prompt
    assert "A) True" in prompt
    assert "B) False" in prompt


def test_estimate_uncertainty(p_true_estimator, mock_samples):
    uncertainty = p_true_estimator.estimate_uncertainty(
        mock_samples, question="Test question?"
    )
    gt_uncertainty = torch.prod(torch.exp(p_true_estimator.llm.probs)).item()
    assert uncertainty == gt_uncertainty
