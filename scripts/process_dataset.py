import json
import argparse
from pathlib import Path
from typing import Dict, List
from loguru import logger
from datasets import load_dataset


def process_coqa(input_file: Path) -> Dict[str, List[Dict[str, str]]]:
    """Process COQA dataset into standardized format."""
    result = {}
    coqa = load_dataset("coqa")
    for split in ["train", "validation"]:
        cur_data = coqa[split]
        result[split] = []
        for sample in cur_data:
            # concatenate story and question
            for question, answer in zip(
                sample["questions"], sample["answers"]["input_text"]
            ):
                question = f'{sample["story"]}\n{question}'
                result[split].append(
                    {
                        "question": question,
                        "gt_answer": answer,
                    }
                )
    return result


def process_nq(input_file: Path) -> Dict[str, List[Dict[str, str]]]:
    """Process Natural Questions dataset into standardized format."""
    pass


def process_bioasq(input_file: Path) -> Dict[str, List[Dict[str, str]]]:
    """Process BioASQ dataset into standardized format."""
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Process QA datasets into standardized format"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["coqa", "nq", "bioasq"],
        help="Dataset to process",
    )
    parser.add_argument("--input", type=Path, required=False, help="Input file path")
    parser.add_argument("--output", type=Path, required=True, help="Output file path")
    args = parser.parse_args()

    # Select processing function based on dataset
    processors = {"coqa": process_coqa, "nq": process_nq, "bioasq": process_bioasq}

    if args.dataset not in processors:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    logger.info(f"Processing {args.dataset} dataset...")
    processed_data = processors[args.dataset](args.input)

    # Save processed data
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(processed_data, f, indent=2)
    logger.info(f"Processed data saved to {args.output}")


if __name__ == "__main__":
    main()
