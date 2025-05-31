import argparse
from loguru import logger
import json
from luq.evals.accuracy_eval import AccuracyEvaluator


def evaluate_dataset(
    input_file: str, output_file: str, model_name: str, model_type: str
):
    """
    Evaluates the accuracy of generated answers in a dataset and saves the results.

    This function loads a dataset from a JSON file, evaluates the accuracy of generated
    answer samples against ground truth answers using the provided model configuration,
    adds the accuracy results to the dataset, and saves the updated dataset to a new file.

    Args:
        input_file (str): Path to the input JSON file containing the dataset. The dataset must
            follow the structure:
            {
                "data": {
                    "<split_name>": [
                        {
                            "samples": [<predicted_answer_1>, <predicted_answer_2>, ...],
                            "gt_answer": <ground_truth_answer>
                        },
                        ...
                    ]
                }
            }
        output_file (str): Path to the output file where the evaluated dataset will be saved.
        model_name (str): The name of the model used for evaluation.
        model_type (str): The type of model being evaluated (e.g., "openai", "hf").

    Raises:
        IOError: If there is an error reading the input file or writing the output file.
    """

    # Initialize evaluator
    evaluator = AccuracyEvaluator(model_name=model_name, model_type=model_type)

    # Load dataset
    try:
        with open(input_file, "r") as f:
            dataset = json.load(f)
    except IOError as e:
        logger.error(f"Error loading input file {input_file}: {e}")
        raise

    all_data = dataset["data"]
    for split, data in all_data.items():
        for example in data:
            predictions = example["samples"]
            ground_truth = example["gt_answer"]

            # Get accuracy scores for each sample
            results = evaluator.evaluate_dataset(
                predictions, [ground_truth] * len(predictions)
            )

            # Add accuracy scores to the example
            example["accuracy"] = results["mean_accuracy"]
            example["individual_accuracies"] = results["individual_scores"]

    # Save updated dataset
    try:
        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=2)
        logger.info(f"Saved evaluated dataset to {output_file}")
    except IOError as e:
        logger.error(f"Error saving to output file {output_file}: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate accuracy of generated answers."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input JSON file with generated samples",
    )
    parser.add_argument(
        "--output-file", type=str, required=True, help="Path to save evaluated dataset"
    )
    parser.add_argument(
        "--model-name", type=str, default="gpt2", help="Model to use for evaluation"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="huggingface",
        choices=["huggingface", "openai"],
        help="Type of model to use",
    )

    args = parser.parse_args()
    evaluate_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model_name,
        model_type=args.model_type,
    )
