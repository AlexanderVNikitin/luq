import pytest
from unittest.mock import patch, MagicMock, mock_open
from scripts.upload_dataset import parse_args, load_json_dataset, upload_to_hub, main
import json
from scripts.add_generations_to_dataset import (
    read_questions_and_answers,
    generate_samples,
    write_samples_to_file,
)


def test_parse_args(monkeypatch):
    test_args = ["script.py", "--path", "dataset.json", "--repo-id", "user/dataset"]
    monkeypatch.setattr("sys.argv", test_args)
    args = parse_args()
    assert args.path == "dataset.json"
    assert args.repo_id == "user/dataset"
    assert args.token is None  # Unless HF_TOKEN is set


def test_load_json_dataset():
    with patch("scripts.upload_dataset.GenerationDataset") as mock_dataset:
        mock_instance = mock_dataset.return_value
        dataset = load_json_dataset("dataset.json")
        mock_dataset.assert_called_once_with("dataset.json")
        assert dataset == mock_instance


def test_upload_to_hub():
    mock_dataset = MagicMock()
    with patch("scripts.upload_dataset.login") as mock_login:
        upload_to_hub(mock_dataset, "user/dataset", "fake_token")
        mock_login.assert_called_once_with(token="fake_token")
        mock_dataset.push_to_hub.assert_called_once_with(
            "user/dataset", private=True, commit_message="Upload initial dataset"
        )


def test_main():
    with patch("scripts.upload_dataset.parse_args") as mock_parse_args, patch(
        "scripts.upload_dataset.os.path.exists", return_value=True
    ) as mock_exists, patch(
        "scripts.upload_dataset.load_json_dataset"
    ) as mock_load_dataset, patch(
        "scripts.upload_dataset.upload_to_hub"
    ) as mock_upload, patch(
        "scripts.upload_dataset.logger.info"
    ) as mock_logger:
        mock_parse_args.return_value = MagicMock(
            path="dataset.json", repo_id="user/dataset", token="fake_token"
        )
        mock_load_dataset.return_value = MagicMock()

        main()

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

    samples = generate_samples(mock_generator, "Test prompt", 1, 1.0, 0.9, 50)
    assert len(samples) == 1
    assert samples[0] == "Sample response"


def test_generate_samples_failure():
    mock_generator = MagicMock(side_effect=Exception("Generation error"))
    with pytest.raises(Exception, match="Generation error"):
        generate_samples(mock_generator, "Test prompt", 1, 1.0, 0.9, 50)


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
