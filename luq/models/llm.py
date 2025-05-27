from dataclasses import dataclass
import typing as T
import torch
import functools
import httpx
import transformers
from openai import OpenAI
from anthropic import Anthropic


@dataclass
class LLMOutput:
    answer: str
    logprobs: torch.Tensor | None = None  # list of logprobs


@dataclass
class LLMSamples:
    samples: T.List[LLMOutput]
    answer: LLMOutput
    params: T.Dict[str, T.Any]

    def __len__(self):
        return len(self.samples)


class LLMWrapper:
    def __call__(self, *args, **kwargs) -> LLMOutput:
        """
        Return a response of an LLM
        """
        raise NotImplementedError("__call__ should be implemented for your LLM")


class HFLLMWrapper(LLMWrapper):
    def __init__(
        self, tokenizer: transformers.AutoTokenizer, model: transformers.PreTrainedModel
    ):
        if isinstance(tokenizer, transformers.PreTrainedTokenizerBase):
            self.tokenizer = tokenizer
        else:
            raise ValueError("Requires a text generation pipeline from transformers")
        if isinstance(model, transformers.PreTrainedModel):
            self.model = model
        else:
            raise ValueError("Requires a text generation pipeline from transformers")

    def __call__(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_new_tokens=1024,
        *args,
        **kwargs
    ) -> LLMOutput:
        """
        Return a response of an LLM
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                temperature=1.0,
            )
        generated_ids = outputs.sequences[0]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        token_scores = outputs.scores
        generated_tokens = generated_ids[len(inputs["input_ids"][0]) :]

        logprobs = []
        for i, token_id in enumerate(generated_tokens):
            logits = token_scores[i][0]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_logprob = log_probs[token_id].item()
            logprobs.append((self.tokenizer.decode([token_id]), token_logprob))

        # logprobs is a list of pairs (token, logprob)
        logprobs = [el[1] for el in logprobs]
        return LLMOutput(
            answer=generated_text,
            logprobs=torch.tensor(logprobs, device=self.model.device),
        )


def update_base_url(request: httpx.Request, openai_endpoint_url: str) -> None:
    if request.url.path == "/chat/completions":
        request.url = request.url.copy_with(path=openai_endpoint_url)


class AzureCustomGPT4Wrapper:
    def __init__(self, openai_endpoint_url, api_key):
        self.openai_endpoint_url = openai_endpoint_url
        self.client = OpenAI(
            base_url=openai_endpoint_url,
            api_key=False,
            default_headers={
                "Ocp-Apim-Subscription-Key": api_key,
            },
            http_client=httpx.Client(
                event_hooks={"request": [functools.partial(update_base_url, openai_endpoint_url=openai_endpoint_url)]}
            ),
        )

    def __call__(self, input: str) -> LLMOutput:
        kwargs = {
            "model": "no_effect",  # Replace with your actual deployment name
            "logprobs": True,
            "messages": [{"role": "user", "content": input}],
            "top_logprobs": 5,
        }
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        token_logprobs = response.choices[0].logprobs.token_logprobs
        if token_logprobs is not None:
            logprobs_tensor = torch.tensor(token_logprobs, dtype=torch.float32)
        else:
            logprobs_tensor = None
        return LLMOutput(answer=content, logprobs=logprobs_tensor)


class ClaudeWrapper:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

    def __call__(self, prompt: str, model: str = "claude-3-opus-20240229", temperature=1.0, max_tokens: int = 1024) -> LLMOutput:
        """
        Generates a response from Claude with optional logprobs.
        """
        # Anthropic API does not currently support logprobs in the chat API.
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )

            # Extract text response
            text = response.content[0].text if response.content else ""
            return LLMOutput(answer=text)

        except Exception as e:
            raise RuntimeError(f"Claude API call failed: {e}")


class BatchLLMWrapper:
    def __call__(self, *args, **kwargs) -> T.List[LLMOutput]:
        """
        Return a response of an LLM
        """
        raise NotImplementedError("__call__ should be implemented for your LLM")


def all_logits_present(samples: T.List[LLMOutput]) -> bool:
    return all(sample.logprobs is not None for sample in samples)


def generate_n_samples_and_answer(
    llm: LLMWrapper,
    prompt: str,
    temp_gen: float = 1.0,
    temp_answer: float = 0.1,
    top_p_gen: float = 0.9,
    top_k_gen: float = 16,
    top_p_ans: float = 0.7,
    top_k_ans: float = 4,
    n_samples: int = 10,
) -> LLMSamples:
    if isinstance(llm, LLMWrapper):
        sampled_answers = [
            llm(prompt, temperature=temp_gen, top_p=top_p_gen, top_k=top_k_gen)
            for _ in range(n_samples)
        ]
        answer = llm(prompt, temperature=temp_gen, top_p=top_p_gen, top_k=top_k_gen)
        params = {
            "prompt": prompt,
            "temp_gen": temp_gen,
            "temp_answer": temp_answer,
            "top_p_gen": top_p_gen,
            "top_k_gen": top_k_gen,
            "top_p_ans": top_p_ans,
            "top_k_ans": top_k_ans,
            "n_samples": n_samples,
            "llm": str(llm),
        }
        return LLMSamples(samples=sampled_answers, answer=answer, params=params)
    else:
        raise NotImplementedError(
            "generation is currently supported only for LLMWrapper"
        )
