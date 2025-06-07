import pytest
import torch
from unittest.mock import MagicMock
from luq.models import HFLLMWrapper, LLMOutput
import transformers


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=transformers.PreTrainedTokenizerBase)

    # Mock the object returned by tokenizer(prompt, return_tensors="pt")
    encoding = MagicMock()
    encoding["input_ids"] = torch.tensor([[0, 1, 2]])
    encoding.to = MagicMock(return_value=encoding)  # important fix

    tokenizer.return_value = encoding
    tokenizer.decode.side_effect = lambda ids, skip_special_tokens=True: (
        "generated text" if skip_special_tokens else str(ids)
    )

    return tokenizer


@pytest.fixture
def mock_model():
    model = MagicMock(spec=transformers.PreTrainedModel)
    model.device = torch.device("cpu")
    
    generated_ids = torch.tensor([[0, 1]])
    scores = [torch.randn(1, 12) for _ in range(2)]
    model.generate = MagicMock()
    model.generate.return_value = MagicMock(
        sequences=generated_ids,
        scores=scores
    )
    return model


def test_valid_initialization(mock_tokenizer, mock_model):
    wrapper = HFLLMWrapper(mock_tokenizer, mock_model)
    assert wrapper.tokenizer == mock_tokenizer
    assert wrapper.model == mock_model


def test_invalid_tokenizer():
    with pytest.raises(ValueError):
        HFLLMWrapper("not_a_tokenizer", MagicMock(spec=transformers.PreTrainedModel))


def test_invalid_model():
    with pytest.raises(ValueError):
        HFLLMWrapper(MagicMock(spec=transformers.PreTrainedTokenizerBase), "not_a_model")


def test_call_method(mock_tokenizer, mock_model):
    wrapper = HFLLMWrapper(mock_tokenizer, mock_model)
    
    # Call the wrapper
    output = wrapper("Hello world")
    
    assert isinstance(output, LLMOutput)
    assert isinstance(output.answer, str)
    assert isinstance(output.logprobs, torch.Tensor)
    assert output.logprobs.ndim == 1