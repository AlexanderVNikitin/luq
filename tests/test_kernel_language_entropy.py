import pytest
import torch
from luq.models.kernel_language_entropy import KernelLanguageEntropyEstimator
from luq.llm import LLMSamples, LLMOutput
from luq.models.kernel_utils import KernelType
from luq.llm.nli import NLIWrapper


@pytest.fixture
def kernel_estimator():
    return KernelLanguageEntropyEstimator()


@pytest.fixture
def mock_samples():
    samples = [
        LLMOutput(answer="Sample 1", logprobs=torch.tensor([0.1, 0.2])),
        LLMOutput(answer="Sample 2", logprobs=torch.tensor([0.2, 0.3])),
        LLMOutput(answer="Sample 3", logprobs=torch.tensor([0.3, 0.4])),
    ]
    return LLMSamples(samples=samples, answer=samples[0], params={})


@pytest.fixture
def mock_nli():
    class MockNLI(NLIWrapper):
        def compare(self, text1, text2):
            return 0.5  # Mock similarity score

    return MockNLI()


def test_get_kernel_validation(kernel_estimator, mock_samples):
    with pytest.raises(
        ValueError,
        match="Either `kernel_type` or `construct_kernel` should be specified",
    ):
        kernel_estimator.get_kernel(mock_samples)

    with pytest.raises(
        ValueError,
        match="Only one of `kernel_type` and `construct_kernel` should be specified",
    ):
        kernel_estimator.get_kernel(
            mock_samples,
            kernel_type=KernelType.HEAT,
            construct_kernel=lambda x: torch.eye(3),
        )


def test_custom_kernel(kernel_estimator, mock_samples):
    def custom_kernel(samples):
        return torch.eye(len(samples.samples))

    kernel = kernel_estimator.get_kernel(mock_samples, construct_kernel=custom_kernel)
    assert isinstance(kernel, torch.Tensor)
    assert kernel.shape == (3, 3)
    assert torch.allclose(kernel, torch.eye(3) / 3)


def test_estimate_uncertainty_validation(kernel_estimator, mock_samples, mock_nli):
    with pytest.raises(
        ValueError, match="Either `nli_model` or `nli_table` should be provided"
    ):
        kernel_estimator.estimate_uncertainty(mock_samples)

    with pytest.raises(
        ValueError, match="Only one of `nli_model` and `nli_table` should be provided"
    ):
        kernel_estimator.estimate_uncertainty(
            mock_samples, nli_model=mock_nli, nli_table={}
        )


def test_compute_entropy(kernel_estimator):
    kernel = torch.eye(3) / 3
    entropy = kernel_estimator.compute_entropy(kernel, normalize=True)
    assert entropy > 1e-6

    uniform_kernel = torch.ones((2, 2)) / 2
    entropy = kernel_estimator.compute_entropy(uniform_kernel, normalize=True)
    assert abs(entropy) < 1e-6
