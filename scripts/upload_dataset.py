from luq.datasets import GenerationDataset
from huggingface_hub import login
import argparse
import os
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub")
    parser.add_argument(
        "--path", type=str, required=True, help="Path to the dataset file"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID on Hugging Face (format: username/dataset-name)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face API token",
    )
    return parser.parse_args()


def load_json_dataset(file_path):
    """Load dataset from JSON file."""
    dataset = GenerationDataset(file_path)
    return dataset


def upload_to_hub(dataset, repo_id, token):
    """Upload dataset to Hugging Face Hub."""
    # Login to Hugging Face
    login(token=token)

    # Push to hub
    dataset.push_to_hub(repo_id, private=True, commit_message="Upload initial dataset")


def main():
    args = parse_args()
    if args.token is None:
        raise ValueError(
            "HF token should be provided as --token or in HF_TOKEN environment variable"
        )

    # Check if file exists
    if not os.path.exists(args.path):
        raise FileNotFoundError(f"Dataset file not found at {args.path}")

    # Load the dataset
    logger.info(f"Loading dataset from {args.path}")
    dataset = load_json_dataset(args.path)

    # Upload to Hugging Face Hub
    logger.info(f"Uploading dataset to {args.repo_id}")
    upload_to_hub(dataset, args.repo_id, args.token)

    logger.info("Dataset uploaded successfully!")


if __name__ == "__main__":
    main()
