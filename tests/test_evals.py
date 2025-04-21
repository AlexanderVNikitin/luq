import pytest
from unittest.mock import patch, MagicMock
from luq.evals.accuracy_eval import AccuracyEvaluator
from luq.evals.uncertainy_eval import UncertaintyEvaluator
from luq.models.base_uq_model import BaseUQModel
from luq.datasets import GenerationDataset


def test_accuracy_evaluator_init():
    # Test HuggingFace initialization
    with patch("luq.evals.accuracy_eval.pipeline") as mock_pipeline:
        AccuracyEvaluator("test-model", "huggingface")
        mock_pipeline.assert_called_once_with("text-classification", model="test-model")

    # Test OpenAI initialization without API key
    with pytest.raises(ValueError):
        AccuracyEvaluator("gpt-3.5-turbo", "openai")

    # Test invalid model type
    with pytest.raises(ValueError):
        AccuracyEvaluator("model", "invalid_type")


def test_accuracy_evaluator_evaluate_answer():
    # Test HuggingFace evaluation
    with patch("luq.evals.accuracy_eval.pipeline") as mock_pipeline:
        mock_model = MagicMock()
        mock_model.return_value = [{"score": 0.8}]
        mock_pipeline.return_value = mock_model
        evaluator = AccuracyEvaluator("test-model", "huggingface")
        score = evaluator.evaluate_answer("prediction", "ground truth")

        assert score == 0.8
        mock_model.assert_called_once()

    # Test OpenAI evaluation
    with patch("luq.evals.accuracy_eval.openai.Completion.create") as mock_completion:
        mock_completion.return_value.choices[0].text = "0.9"

        with patch("luq.evals.accuracy_eval.openai.api_key", "fake-key"):
            evaluator = AccuracyEvaluator("gpt-3.5-turbo", "openai")
            score = evaluator.evaluate_answer("prediction", "ground truth")

            assert score == 0.9
            mock_completion.assert_called_once()


def test_accuracy_evaluator_evaluate_dataset():
    evaluator = AccuracyEvaluator("gpt2", "huggingface")

    with patch.object(evaluator, "evaluate_answer", return_value=0.8):
        predictions = ["pred1", "pred2"]
        ground_truths = ["truth1", "truth2"]

        results = evaluator.evaluate_dataset(predictions, ground_truths)

        assert "mean_accuracy" in results
        assert "individual_scores" in results
        assert len(results["individual_scores"]) == 2
        assert results["mean_accuracy"] == 0.8


def test_uncertainty_evaluator_evaluate_auroc():
    mock_model = MagicMock(spec=BaseUQModel)
    mock_model.estimate_uncertainty.return_value = 0.5

    evaluator = UncertaintyEvaluator(mock_model)

    mock_dataset = MagicMock(spec=GenerationDataset)
    mock_dataset.__iter__.return_value = [
        {"question": "q1", "samples": ["s1"], "accuracy": 1.0},
        {"question": "q2", "samples": ["s2"], "accuracy": 0.0},
    ]

    results = evaluator.evaluate_auroc(mock_dataset)

    assert "auroc" in results
    assert "mean_uncertainty" in results
    assert "std_uncertainty" in results
    assert "mean_accuracy" in results


def test_uncertainty_evaluator_evaluate_threshold_curve():
    mock_model = MagicMock(spec=BaseUQModel)
    mock_model.estimate_uncertainty.return_value = 0.5

    evaluator = UncertaintyEvaluator(mock_model)

    mock_dataset = MagicMock(spec=GenerationDataset)
    mock_dataset.__iter__.return_value = [
        {"question": "q1", "samples": ["s1"], "accuracy": 1.0},
        {"question": "q2", "samples": ["s2"], "accuracy": 0.0},
    ]

    results = evaluator.evaluate_threshold_curve(mock_dataset)

    assert "autc" in results
    assert "thresholds" in results
    assert "coverage_values" in results
    assert "accuracy_values" in results


def test_uncertainty_evaluator_evaluate_threshold():
    mock_model = MagicMock(spec=BaseUQModel)
    mock_model.estimate_uncertainty.return_value = 0.5

    evaluator = UncertaintyEvaluator(mock_model)

    mock_dataset = MagicMock(spec=GenerationDataset)
    mock_dataset.__iter__.return_value = [
        {"question": "q1", "samples": ["s1"], "accuracy": 1.0},
        {"question": "q2", "samples": ["s2"], "accuracy": 0.0},
    ]

    results = evaluator.evaluate_threshold(mock_dataset, uncertainty_threshold=0.7)

    assert "low_uncertainty_accuracy" in results
    assert "high_uncertainty_accuracy" in results
    assert "low_uncertainty_fraction" in results
