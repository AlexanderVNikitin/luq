import os
import argparse
from pathlib import Path
import json
from loguru import logger
from datasets import load_dataset, DatasetDict

from luq.models import PredictiveEntropyEstimator
from luq.evals.uncertainy_eval import UncertaintyEvaluator
from luq.datasets import GenerationDataset


def evaluate_model(model, dataset, model_name):
    """Evaluate a single uncertainty quantification model."""
    evaluator = UncertaintyEvaluator(model)

    auroc_results = evaluator.evaluate_auroc(dataset)
    threshold_curve_results = evaluator.evaluate_threshold_curve(dataset)
    threshold_results = evaluator.evaluate_threshold(dataset, uncertainty_threshold=0.5)

    return {
        "model": model_name,
        "auroc": auroc_results["auroc"],
        "mean_uncertainty": auroc_results["mean_uncertainty"],
        "std_uncertainty": auroc_results["std_uncertainty"],
        "mean_accuracy": auroc_results["mean_accuracy"],
        
        "autc": threshold_curve_results["autc"],
        "acc_thresholds": threshold_curve_results["thresholds"],
        "acc_coverages": threshold_curve_results["coverage_values"],
        "accuracy_values": threshold_curve_results["accuracy_values"],

        "low_uncertainty_accuracy": threshold_results["low_uncertainty_accuracy"],
        "high_uncertainty_accuracy": threshold_results["high_uncertainty_accuracy"],
        "low_uncertainty_fraction": threshold_results["low_uncertainty_fraction"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate uncertainty quantification methods"
    )
    parser.add_argument(
        "--dataset-path", type=str, required=True, help="Path to the processed dataset"
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Path to save evaluation results"
    )
    args = parser.parse_args()

    if os.path.exists(args.dataset_path):
        logger.info(f"Loading from file: {args.dataset_path}")
        dataset = GenerationDataset.load_from_json(args.dataset_path)
    else:
        logger.info(f"Loading from HF: {args.dataset_path}")
        dataset = load_dataset(args.dataset_path)

    # Initialize models
    models = {
        # "semantic_entropy": SemanticEntropyEstimator(),
        "predictive_entropy": PredictiveEntropyEstimator(),
    }

    # Evaluate each model
    results = []
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        model_results = {split: evaluate_model(model, dataset[split], model_name) for split in dataset.keys()}
        results.append(model_results)

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
