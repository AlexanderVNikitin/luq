import numpy as np
from typing import List, Dict, Union
from transformers import pipeline
import openai
from loguru import logger

from enum import Enum


class ModelType(Enum):
    huggingface: str = "huggingface"
    openai: str = "openai"


class AccuracyEvaluator:
    """Evaluates accuracy of LLM answers against ground truth."""

    def __init__(
        self,
        model_name: str = "gpt2",
        model_type: ModelType = ModelType.huggingface,
        prompt_template: str | None = None,
    ):
        """
        Initialize the evaluator.

        Args:
            model_name: Name/path of the model to use for evaluation
            model_type: Either "huggingface" or "openai"
        """
        self.model_name = model_name
        self.model_type = model_type.lower()
        self.prompt_template = prompt_template
        if self.prompt_template is None:
            self.prompt_template = """Rate how similar these two answers are on a scale of 0 to 1:
            Answer 1: {{predicted}}
            Answer 2: {{ground_truth}}
            Score:"""

        if not isinstance(self.model_type, ModelType):
            self.model_type = ModelType(self.model_type)

        if self.model_type == ModelType.huggingface:
            try:
                self.model = pipeline("text-classification", model=model_name)
            except Exception as e:
                logger.error(f"Failed to load HuggingFace model {model_name}: {e}")
                raise
        elif self.model_type == ModelType.openai:
            if not openai.api_key:
                raise ValueError("OpenAI API key not found")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def construct_prompt(self, predicted: str, ground_truth: str) -> str:
        return self.prompt_template.format(
            predicted=predicted, ground_truth=ground_truth
        )

    def evaluate_answer(self, predicted: str, ground_truth: str) -> float:
        """
        Evaluate a single predicted answer against ground truth.

        Args:
            predicted: The model's predicted answer
            ground_truth: The ground truth answer

        Returns:
            Accuracy score between 0 and 1
        """
        prompt = self.construct_prompt(predicted, ground_truth)

        if self.model_type == ModelType.huggingface:
            result = self.model(prompt)[0]
            return float(result["score"])

        elif self.model_type == ModelType.openai:
            response = openai.Completion.create(
                model=self.model_name, prompt=prompt, max_tokens=1, temperature=0.1
            )
            return float(response.choices[0].text.strip())

    def evaluate_dataset(
        self, predictions: List[str], ground_truths: List[str]
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Evaluate a list of predictions against ground truths.

        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers

        Returns:
            Dict containing mean accuracy and per-example scores
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions and ground truths must match")

        scores = []
        for pred, truth in zip(predictions, ground_truths):
            score = self.evaluate_answer(pred, truth)
            scores.append(score)

        return {"mean_accuracy": float(np.mean(scores)), "individual_scores": scores}
