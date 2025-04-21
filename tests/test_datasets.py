import pytest
import json
import tempfile
from datasets import DatasetDict
from luq.datasets import GenerationDataset


@pytest.fixture
def temp_json_file():
    """
    Creates a temporary JSON file for testing.
    """
    test_data = {
        "llm": "test_llm",
        "raw_dataset": "test_dataset",
        "data": {
            "train": [
                {
                    "question": "Is this a test question?",
                    "samples": ["Yes", "No"],
                    "answer": "Yes",
                    "samples_temp": 1.0,
                    "answer_temp": 0.1,
                    "top-p": 0.9,
                    "gt_answer": "Yes",
                }
            ]
            * 10,
            "test": [
                {
                    "question": "Is this a test question?",
                    "samples": ["Yes", "No"],
                    "answer": "Yes",
                    "samples_temp": 1.0,
                    "answer_temp": 0.1,
                    "top-p": 0.9,
                    "gt_answer": "Yes",
                }
            ]
            * 10,
        },
    }

    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8")
    json.dump(test_data, temp_file)
    temp_file.close()

    yield temp_file.name

    import os

    os.remove(temp_file.name)


@pytest.fixture
def dataset(temp_json_file):
    return GenerationDataset(temp_json_file)


def test_load_dataset(dataset):
    """
    Tests that the dataset loads correctly.
    """
    assert isinstance(dataset, DatasetDict)
    assert len(dataset) == 2
    assert list(dataset.keys()) == ["train", "test"]
    assert dataset["train"][0]["question"] == "Is this a test question?"
    assert dataset["test"][0]["question"] == "Is this a test question?"


def test_split_dataset(dataset):
    """
    Tests the dataset splitting function.
    """
    split_data = dataset["train"].train_test_split(train_size=0.5)
    assert isinstance(split_data, DatasetDict)
    assert "train" in split_data
    assert "test" in split_data
    assert len(split_data["train"]) == 5
    assert len(split_data["test"]) == 5
