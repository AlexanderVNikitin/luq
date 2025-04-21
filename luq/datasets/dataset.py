from datasets import Dataset, DatasetDict
import json
from typing import Dict


class GenerationDataset(DatasetDict):
    def __init__(self, data_path: str = None, arrow_table=None):
        """
        Initializes the dataset object.
        :param data_path: Path to the JSON file containing the dataset.
        """
        if data_path is not None:
            dataset_dict = self.load_from_json(data_path)
            super().__init__(dataset_dict)
        elif arrow_table is not None:
            dataset = Dataset(arrow_table)
            super().__init__({"train": dataset})
        else:
            raise ValueError("Either data_path or arrow_table must be provided")

    @staticmethod
    def load_from_json(data_path: str) -> Dataset:
        """
        Loads the dataset from a JSON file and converts it into a Hugging Face Dataset object.
        """
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        split_datasets = {}
        for split, items in raw_data["data"].items():
            processed_items = [
                {
                    "question": item["question"],
                    "samples": item["samples"],
                    "logprobs": item.get("logprobs", []),
                    "answer": item["answer"],
                    "gt_answer": item["gt_answer"],
                    "accuracy": item.get("accuracy"),
                }
                for item in items
            ]
            split_datasets[split] = Dataset.from_list(processed_items)
        return DatasetDict(split_datasets)

    def split_dataset(self, train_size: float = 0.8) -> Dict[str, "GenerationDataset"]:
        """
        Splits the dataset into train and test sets.
        :param train_size: Proportion of the dataset to include in the train split.
        :return: Dictionary containing train and test GenerationDataset objects
        """
        splits = super().train_test_split(train_size=train_size)
        return {
            "train": GenerationDataset(arrow_table=splits["train"]._data),
            "test": GenerationDataset(arrow_table=splits["test"]._data),
        }

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> "GenerationDataset":
        """
        Creates a GenerationDataset from a regular Dataset object
        """
        return cls(arrow_table=dataset._data)
