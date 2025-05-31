from loguru import logger
import typing as T
from luq.models import LLMSamples, LLMWrapper, LLMOutput
from luq.methods.base_uq_model import BaseUQModel
from luq.utils import SeqProbMode


class PTrueEstimator(BaseUQModel):
    """Uncertainty estimator based on P(true) metric.

    This estimator constructs a prompt that asks an LLM to assess the truthfulness of a given
    answer compared to brainstormed alternatives, then uses the model's response to estimate
    uncertainty.
    """

    def __init__(self, llm: LLMWrapper):
        """Initialize the P(true) estimator.

        Args:
            llm (LLMWrapper): Wrapper around a language model used for generating predictions.
        """
        super().__init__()
        self.llm = llm

    def construct_p_true_prompt(
        self,
        question: str,
        most_probable_answer: str,
        brainstormed_answers: T.List[str],
        hint: bool = False,
    ) -> str:
        """Construct a prompt for the P(true) uncertainty estimation task.

        This prompt asks the language model to evaluate whether the most probable answer
        aligns with a set of brainstormed answers.

        Args:
            question (str): The original input question.
            most_probable_answer (str): The most likely answer as determined by the model.
            brainstormed_answers (List[str]): Other generated answers used for comparison.
            hint (bool, optional): If True, include more explicit instructions in the prompt.
                Defaults to False.

        Returns:
            str: A formatted prompt string suitable for LLM input.
        """
        prompt = f"Question: {question}\nBrainstormed Answers: "
        for answer in brainstormed_answers + [most_probable_answer]:
            prompt += f"{answer.strip()}\n"
        prompt += f"Possible answer: {most_probable_answer}\n"

        if not hint:
            prompt += "Is the possible answer:\n"
            prompt += "A) True\n"
            prompt += "B) False\n"
            prompt += "The possible answer is:"
        else:
            prompt += "Do the brainstormed answers match the possible answer? Respond with A if they do, if they do not respond with B. Answer:"

        return prompt

    def estimate_uncertainty(
        self,
        samples: LLMSamples,
        most_probable_answer: LLMOutput | None = None,
        hint: bool = False,
        **kwargs,
    ) -> float:
        """Estimate uncertainty using the P(true) approach.

        The method selects the most probable answer from samples and constructs a prompt
        to ask the model whether that answer is supported by alternative answers (brainstormed).
        The returned uncertainty is derived from the model's belief in the answer's truthfulness.

        Args:
            samples (LLMSamples): Samples of generated answers from the language model.
            most_probable_answer (LLMOutput, optional): Precomputed most probable answer. If not provided,
                it will be inferred from the samples.
            hint (bool, optional): If True, include hint-style prompt formatting. Defaults to False.
            **kwargs: Additional keyword arguments. Expects `question` (str) to be passed here.

        Returns:
            float: The 1 - P(true) score. Higher values indicate greater uncertainty; lower values imply less uncertainty.
        """
        if most_probable_answer is None:
            # Get most probable answer (answer with highest logprobs)
            most_probable_answer = max(
                samples.samples,
                key=lambda x: self.compute_sequence_probability(x.logprobs)
                if x.logprobs is not None
                else float("-inf"),
            )

        # Get brainstormed answers excluding most probable
        brainstormed = [
            s.answer for s in samples.samples if s.answer != most_probable_answer.answer
        ]

        # Construct P(true) prompt
        prompt = self.construct_p_true_prompt(
            kwargs.get("question", ""),
            most_probable_answer.answer,
            brainstormed,
            hint=hint,
        )

        # Get P(true) prediction
        response = self.llm(prompt, temperature=0.1)

        # Calculate negative log likelihood of response
        if response.answer == "A":
            return 1 - self.compute_sequence_probability(
                response.logprobs, seq_prob_mode=SeqProbMode.PROD
            )
        elif response.answer == "B":
            return self.compute_sequence_probability(
                response.logprobs, seq_prob_mode=SeqProbMode.PROD
            )
        else:
            logger.error(
                f"LLM has responded {response.answer}, should be either A or B. Return p(true) = 0"
            )
            return 1
