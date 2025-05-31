import argparse
import json
import torch
from tqdm import tqdm
from loguru import logger
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Dict, List

import luq.utils


def read_questions_and_answers(file_path: str) -> Dict:
    """Reads questions and answers from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing questions and answers.

    Returns:
        list or dict: The parsed content of the JSON file, typically a list or dictionary,
        depending on the structure of the JSON.

    Raises:
        IOError: If the file cannot be read or the contents cannot be parsed as JSON.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        logger.info(f"Successfully read {len(data)} question(s) from {file_path}.")
        return data
    except IOError as e:
        logger.error(f"Failed to read the file {file_path}: {e}")
        raise


def generate_samples(
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        prompt: str,
        num_samples: int,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> List[str]:
    """Generate text samples one by one using a Hugging Face model and tokenizer.

    This function generates `num_samples` sequences conditioned on the input
    `prompt`, using sampling with temperature, nucleus sampling (top_p), and top-k
    filtering. It processes each sample individually to reduce memory usage.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer used to encode and decode text.
        model (PreTrainedModel): Pretrained model used for generation.
        prompt (str): The input text prompt to condition generation on.
        num_samples (int): Number of samples to generate.
        temperature (float): Sampling temperature; higher values increase diversity.
        top_p (float): Cumulative probability for nucleus sampling.
        top_k (int): Number of highest probability tokens to keep for top-k sampling.

    Returns:
        List[str]: A list of generated text samples as strings.

    Raises:
        ValueError: If generation fails due to invalid input or model issues.
    """
    try:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        samples = []

        for _ in range(num_samples):
            output = model.generate(
                input_ids=input_ids,
                max_length=model.config.max_position_embeddings,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            samples.append(decoded)

        logger.info(f"Generated {num_samples} sample(s) for prompt: {prompt}")
        return samples

    except ValueError as e:
        logger.error(f"Error generating samples: {e}")
        raise


def write_samples_to_file(samples: List[str], parameters: Dict, output_file: str) -> None:
    """Write generated samples and their parameters to a JSON file.

    Saves the generated samples along with the generation parameters into
    a JSON file specified by `output_file`. The data is saved under the keys
    "data" for samples and "parameters" for the associated metadata.

    Args:
        samples (list): A list of generated text samples to write.
        parameters (dict): A dictionary of parameters used to generate the samples.
        output_file (str): Path to the output JSON file.

    Raises:
        IOError: If there is an error writing to the file.
    """
    try:
        result = {"data": samples, "parameters": parameters}
        with open(output_file, "w") as file:
            json.dump(result, file, indent=4)
        logger.info(f"Successfully wrote {len(samples)} samples to {output_file}.")
    except IOError as e:
        logger.error(f"Failed to write samples to {output_file}: {e}")
        raise


def main(
    input_file,
    output_file,
    num_samples,
    num_questions,
    model_name,
    temperature,
    temperature_answer,
    top_p,
    top_k,
    torch_dtype=torch.float16,
    device=0, # >=0 for gpu and -1 for cpu
):
    """Main function to handle the workflow."""
    if num_questions == -1:
        num_questions = None
    logger.info("Starting the sample generation process.")
    logger.info(f"Loading {model_name}")
    try:
        logger.info(f"{torch.cuda.device_count()} GPU devices available")
        #generator = pipeline("text-generation", model=model_name, torch_dtype=torch_dtype, devic=device, device_map="auto")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",  # Use bitsandbytes or accelerate to split across GPUs or use CPU+GPU
        )
        logger.info(f"Successfully loaded model {model_name} to {model.device}.")
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise

    data = read_questions_and_answers(input_file)
    processed_data = {}
    for split, cur_data in data.items():
        logger.info(f"Processing split: {split}")
        processed_data[split] = []
        parameters = {
            "temperature": temperature,
            "temperature_answer": temperature_answer,
            "top_p": top_p,
            "top_k": top_k,
            "model_name": model_name,
            "num_samples": num_samples,
        }

        for entry in tqdm(cur_data[:num_questions]):
            question = entry.get("question")
            gt_answer = entry.get("gt_answer")
            if not question:
                logger.warning(f"Skipping entry with missing question. Entry: {entry}")
                continue
            samples = generate_samples(
                tokenizer, model, question, num_samples, temperature, top_p=top_p, top_k=top_k
            )
            answer = generate_samples(
                tokenizer,
                model,
                question,
                num_samples=1,
                temperature=temperature_answer,
                top_p=top_p,
                top_k=top_k,
            )
            processed_data[split].append(
                {
                    "question": question,
                    "samples": samples,
                    "answer": answer,
                    "gt_answer": gt_answer,
                }
            )

    write_samples_to_file(processed_data, parameters, output_file)
    logger.info("Sample generation process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate samples using a Hugging Face model."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input file with questions and answers.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the output file to save samples.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to generate per question.",
    )
    parser.add_argument(
        "--model-name", type=str, default="gpt2", help="Hugging Face model name."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for generating responses.",
    )
    parser.add_argument(
        "--temperature-answer",
        type=float,
        default=0.1,
        help="Temperature for generating final answer.",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=-1,
        help="Number of questions used from the dataset (-1 to use all)",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        help="Torch floating point precision.",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU id (>= 0), or CPU (-1)"
    )

    parser.add_argument("--top-k", type=int, default=25, help="top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.25, help="top-p sampling")

    args = parser.parse_args()

    main(
        input_file=args.input_file,
        output_file=args.output_file,
        num_samples=args.num_samples,
        model_name=args.model_name,
        temperature=args.temperature,
        temperature_answer=args.temperature_answer,
        top_k=args.top_k,
        top_p=args.top_p,
        num_questions=args.num_questions,
        torch_dtype=luq.utils.dtype_map[args.torch_dtype],
        device=args.device
    )
