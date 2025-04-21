import torch

from luq.llm import LLMWrapper, LLMOutput, LLMSamples, generate_n_samples_and_answer


class MockLLM(LLMWrapper):
    def __call__(
        self, prompt: str, top_p: float, top_k: float, temperature: float
    ) -> LLMOutput:
        return LLMOutput(
            f"It is the final answer to prompt <<{prompt}>>",
            logprobs=torch.tensor(
                [-1.0, -12.0, 7.2, -1.0, -12.0, 7.2, -1.0, -12.0, 7.2]
            ),
        )


def test_gen_samples():
    """Test generation of samples of luq.llm.generate_n_samples_and_answer"""
    llm = MockLLM()
    params = {
        "prompt": "test prompt",
        "temp_gen": 0.7,
        "temp_answer": 0.1,
        "top_p_gen": 0.9,
        "top_k_gen": 40,
        "top_p_ans": 0.9,
        "top_k_ans": 40,
        "n_samples": 5,
    }
    samples = generate_n_samples_and_answer(llm=llm, **params)

    params["llm"] = str(llm)
    assert isinstance(samples, LLMSamples)
    assert len(samples.samples) == 5
    assert isinstance(samples.answer, LLMOutput)
    assert isinstance(samples.params, dict)
    assert samples.params == params
