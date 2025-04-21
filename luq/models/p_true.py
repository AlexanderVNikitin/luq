from loguru import logger
import typing as T
from luq.llm import LLMSamples, LLMWrapper, LLMOutput
from luq.models.base_uq_model import BaseUQModel
from luq.utils import SeqProbMode


class PTrueEstimator(BaseUQModel):
    def __init__(self, llm: LLMWrapper):
        """
        Initialize P(true) estimator for uncertainty quantification.

        Args:
            llm: LLM wrapper for making predictions
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
        """Construct prompt for P(true) uncertainty metric."""
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
        """
        Estimate uncertainty using P(true) framework: sampling multiple questions and asking whether a final answer supported by those samples. Then estimate probability that the answer is matched.

        Returns P(true) score as uncertainty measure.
        Lower P(true) indicates higher uncertainty.
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
            return self.compute_sequence_probability(
                response.logprobs, seq_prob_mode=SeqProbMode.PROD
            )
        elif response.answer == "B":
            return 1 - self.compute_sequence_probability(
                response.logprobs, seq_prob_mode=SeqProbMode.PROD
            )
        else:
            logger.error(
                f"LLM has responded {response.answer}, should be either A or B. Return p(true) = 0"
            )
            return 0
