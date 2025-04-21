import torch
import numpy as np
from typing import Dict, List, Union
from sklearn.metrics import roc_auc_score, auc
from loguru import logger
from luq.models.base_uq_model import BaseUQModel
from luq.datasets import GenerationDataset
from luq.llm import LLMSamples, LLMOutput


class UncertaintyEvaluator:
    """Evaluates how well uncertainty estimates predict model accuracy."""

    # TODO separate uncertainty calc from eval

    def __init__(self, uncertainty_model: BaseUQModel):
        """
        Initialize the evaluator.

        Args:
            uncertainty_model: Model that provides uncertainty estimates
        """
        self.uncertainty_model = uncertainty_model

    def evaluate_auroc(self, dataset: GenerationDataset, **kwargs) -> Dict[str, float]:
        """
        Calculate AUROC of using uncertainty to predict accuracy.

        Args:
            dataset: Dataset containing model outputs and accuracies

        Returns:
            Dict containing AUROC score and other metrics
        """
        # Get uncertainty estimates for each example
        uncertainties = []
        accuracies = []

        for entry in dataset:
            samples = [
                LLMOutput(sample, logprobs)
                for sample, logprobs in zip(
                    dataset["samples"], torch.Tensor(dataset["logprobs"])
                )
            ]
            llm_samples = LLMSamples(
                samples=samples, answer=dataset["answer"], params={}
            )  # todo add params
            uncertainty = self.uncertainty_model.estimate_uncertainty(
                question=entry["question"], samples=llm_samples, **kwargs
            )
            uncertainties.append(uncertainty)
            accuracies.append(entry["accuracy"])
        uncertainties = np.array(uncertainties)
        accuracies = np.array(accuracies)

        # Calculate AUROC using uncertainties to predict accuracy
        # Lower uncertainty should predict higher accuracy
        auroc = roc_auc_score((accuracies > 0.5).astype(int), 1 - uncertainties)

        results = {
            "auroc": float(auroc),
            "mean_uncertainty": float(np.mean(uncertainties)),
            "std_uncertainty": float(np.std(uncertainties)),
            "mean_accuracy": float(np.mean(accuracies)),
        }

        logger.info(f"Uncertainty evaluation results: {results}")
        return results

    def evaluate_threshold_curve(
        self, dataset: GenerationDataset, num_thresholds: int = 100
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Calculate area under threshold-accuracy curve.

        Args:
            dataset: Dataset containing model outputs and accuracies
            num_thresholds: Number of threshold points to evaluate

        Returns:
            Dict containing AUTC score and curve points
        """
        try:
            uncertainties = []
            accuracies = []

            for example in dataset:
                samples = [
                    LLMOutput(sample, logprobs)
                    for sample, logprobs in zip(
                        dataset["samples"], torch.Tensor(dataset["logprobs"])
                    )
                ]
                llm_samples = LLMSamples(
                    samples=samples, answer=dataset["answer"], params={}
                )
                uncertainty = self.uncertainty_model.estimate_uncertainty(
                    question=example["question"], samples=llm_samples
                )
                uncertainties.append(uncertainty)
                accuracies.append(example["accuracy"])

            uncertainties = np.array(uncertainties)
            accuracies = np.array(accuracies)

            # Generate thresholds from min to max uncertainty
            thresholds = np.linspace(
                np.min(uncertainties), np.max(uncertainties), num_thresholds
            )

            # Calculate accuracy for each threshold
            coverage_values = []
            accuracy_values = []

            for threshold in thresholds:
                mask = uncertainties <= threshold
                if np.sum(mask) > 0:
                    accuracy = np.mean(accuracies[mask])
                    coverage = np.mean(mask)
                    coverage_values.append(coverage)
                    accuracy_values.append(accuracy)

            # Calculate area under threshold-accuracy curve
            autc = auc(coverage_values, accuracy_values)

            results = {
                "autc": float(autc),
                "thresholds": thresholds.tolist(),
                "coverage_values": coverage_values,
                "accuracy_values": accuracy_values,
            }

            logger.info(f"Threshold curve evaluation results: {results}")
            return results

        except Exception as e:
            logger.error(f"Error evaluating threshold curve: {e}")
            raise

    def evaluate_threshold(
        self, dataset: GenerationDataset, uncertainty_threshold: float
    ) -> Dict[str, float]:
        """
        Evaluate accuracy above/below an uncertainty threshold.

        Args:
            dataset: Dataset containing model outputs and accuracies
            uncertainty_threshold: Threshold for filtering by uncertainty

        Returns:
            Dict containing accuracy metrics for high/low uncertainty examples
        """
        uncertainties = []
        accuracies = []

        for example in dataset:
            samples = [
                LLMOutput(sample, logprobs)
                for sample, logprobs in zip(
                    dataset["samples"], torch.Tensor(dataset["logprobs"])
                )
            ]
            llm_samples = LLMSamples(
                samples=samples, answer=dataset["answer"], params={}
            )
            uncertainty = self.uncertainty_model.estimate_uncertainty(
                question=example["question"], samples=llm_samples
            )
            uncertainties.append(uncertainty)
            accuracies.append(example["accuracy"])

        uncertainties = np.array(uncertainties)
        accuracies = np.array(accuracies)

        # Split into high/low uncertainty
        low_uncertainty_mask = uncertainties < uncertainty_threshold
        high_uncertainty_mask = uncertainties >= uncertainty_threshold

        results = {
            "low_uncertainty_accuracy": float(
                np.mean(accuracies[low_uncertainty_mask])
            ),
            "high_uncertainty_accuracy": float(
                np.mean(accuracies[high_uncertainty_mask])
            ),
            "low_uncertainty_fraction": float(np.mean(low_uncertainty_mask)),
        }

        logger.info(f"Threshold evaluation results: {results}")
        return results
