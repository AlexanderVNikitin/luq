import pytest
from unittest.mock import patch, MagicMock, mock_open
import sys
import tempfile
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from upload_dataset import parse_args, load_json_dataset, upload_to_hub
from upload_dataset import main as main_upload
import json

from add_generations_to_dataset import (
    read_questions_and_answers,
    generate_samples,
    write_samples_to_file,
    main
)


def test_parse_args(monkeypatch):
    test_args = ["script.py", "--path", "dataset.json", "--repo-id", "user/dataset"]
    monkeypatch.setattr("sys.argv", test_args)
    args = parse_args()
    assert args.path == "dataset.json"
    assert args.repo_id == "user/dataset"
    assert args.token is None  # Unless HF_TOKEN is set


def test_load_json_dataset():
    with patch("upload_dataset.GenerationDataset") as mock_dataset:
        mock_instance = mock_dataset.return_value
        dataset = load_json_dataset("dataset.json")
        mock_dataset.assert_called_once_with("dataset.json")
        assert dataset == mock_instance


def test_upload_to_hub():
    mock_dataset = MagicMock()
    with patch("upload_dataset.login") as mock_login:
        upload_to_hub(mock_dataset, "user/dataset", "fake_token")
        mock_login.assert_called_once_with(token="fake_token")
        mock_dataset.push_to_hub.assert_called_once_with(
            "user/dataset", private=True, commit_message="Upload initial dataset"
        )


def test_main():
    with patch("upload_dataset.parse_args") as mock_parse_args, patch(
        "upload_dataset.os.path.exists", return_value=True
    ) as mock_exists, patch(
        "upload_dataset.load_json_dataset"
    ) as mock_load_dataset, patch(
        "upload_dataset.upload_to_hub"
    ) as mock_upload, patch(
        "upload_dataset.logger.info"
    ) as mock_logger:
        mock_parse_args.return_value = MagicMock(
            path="dataset.json", repo_id="user/dataset", token="fake_token"
        )
        mock_load_dataset.return_value = MagicMock()

        main_upload()

        mock_exists.assert_called_once_with("dataset.json")
        mock_load_dataset.assert_called_once_with("dataset.json")
        mock_upload.assert_called_once_with(
            mock_load_dataset.return_value, "user/dataset", "fake_token"
        )
        mock_logger.assert_called()


def test_read_questions_and_answers_success():
    mock_data = json.dumps({"data": [{"question": "What is AI?"}]})
    with patch("builtins.open", mock_open(read_data=mock_data)):
        data = read_questions_and_answers("dummy_path.json")
        assert "data" in data
        assert len(data["data"]) == 1
        assert data["data"][0]["question"] == "What is AI?"


def test_read_questions_and_answers_failure():
    with patch("builtins.open", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            read_questions_and_answers("non_existent_file.json")


def test_generate_samples_success():
    mock_generator = MagicMock()
    mock_generator.return_value = [{"generated_text": "Sample response"}]
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "Sample response"

    samples = generate_samples(mock_tokenizer, mock_generator, "Test prompt", 1, 1.0, 0.9, 50)
    assert len(samples) == 1
    assert samples[0] == "Sample response"


def test_generate_samples_failure():
    mock_generator = MagicMock()
    mock_generator.generate.side_effect = Exception("Generation error")
    mock_tokenizer = MagicMock()
    with pytest.raises(Exception, match="Generation error"):
        generate_samples(mock_tokenizer, mock_generator, "Test prompt", 1, 1.0, 0.9, 50)


def test_write_samples_to_file_success():
    samples = [{"question": "What is AI?", "samples": ["Artificial Intelligence"]}]
    parameters = {"model": "gpt2"}
    mock_file = mock_open()

    with patch("builtins.open", mock_file):
        write_samples_to_file(samples, parameters, "output.json")

    mock_file.assert_called_once_with("output.json", "w")
    handle = mock_file()
    written_data = json.loads(
        "".join(call.args[0] for call in handle.write.call_args_list)
    )
    assert "data" in written_data
    assert "parameters" in written_data
    assert written_data["data"] == samples
    assert written_data["parameters"] == parameters


def test_write_samples_to_file_failure():
    with patch("builtins.open", side_effect=PermissionError):
        with pytest.raises(PermissionError):
            write_samples_to_file([], {}, "output.json")


@pytest.fixture
def mock_data():
    return {
        "train": [
            {"question": "What is the capital of France?", "gt_answer": "Paris"},
            {"question": "What is 2 + 2?", "gt_answer": "4"},
        ]
    }


@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("add_generations_to_dataset.read_questions_and_answers")
@patch("add_generations_to_dataset.write_samples_to_file")
def test_add_generations_to_dataset_main(
    mock_write,
    mock_read,
    mock_model_cls,
    mock_tokenizer_cls,
    mock_data
):
    # Prepare mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_tensors = "pt"
    mock_tokenizer.encode.return_value = [0]
    mock_tokenizer.decode.return_value = "sampled text"
    mock_tokenizer_cls.return_value = mock_tokenizer

    # Prepare mock model
    mock_model = MagicMock()
    mock_model.generate.return_value = [[0, 1, 2]]
    mock_model.device = "cpu"
    mock_model.config.max_position_embeddings = 10
    mock_model_cls.return_value = mock_model

    # Mock read input data
    mock_read.return_value = mock_data

    # Use a temporary output file
    with tempfile.NamedTemporaryFile(suffix=".json") as tmpfile:
        main(
            input_file="dummy_input.json",
            output_file=tmpfile.name,
            num_samples=1,
            num_questions=1,
            model_name="gpt2",
            temperature=1.0,
            temperature_answer=0.1,
            top_p=0.9,
            top_k=25,
            torch_dtype="float16",
            device=-1,
        )

        # Assertions
        assert mock_read.called
        assert mock_tokenizer_cls.called
        assert mock_model_cls.called
        assert mock_write.called

        # Verify structure of the written data
        args, kwargs = mock_write.call_args
        output_data = args[0]  # samples
        output_params = args[1]  # parameters

        assert "train" in output_data
        assert isinstance(output_data["train"], list)
        assert output_params["model_name"] == "gpt2"
