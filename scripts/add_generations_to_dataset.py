import argparse
import json
from tqdm import tqdm
from loguru import logger
from transformers import pipeline


def read_questions_and_answers(file_path):
    """Reads questions and answers from a JSON file."""
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        logger.info(f"Successfully read {len(data)} question(s) from {file_path}.")
        return data
    except Exception as e:
        logger.error(f"Failed to read the file {file_path}: {e}")
        raise


def generate_samples(generator, prompt, num_samples, temperature, top_p, top_k):
    """Generate samples using Hugging Face text generation model."""
    try:
        responses = generator(
            prompt,
            max_length=generator.model.config.max_position_embeddings,
            # max_new_tokens=1024,
            num_return_sequences=num_samples,
            temperature=temperature,
        )
        samples = [resp["generated_text"] for resp in responses]
        logger.info(f"Generated {num_samples} sample(s) for prompt: {prompt}")
        return samples
    except Exception as e:
        logger.error(f"Error generating samples: {e}")
        raise


def write_samples_to_file(samples, parameters, output_file):
    """Write generated samples to a file."""
    try:
        result = {"data": samples, "parameters": parameters}
        with open(output_file, "w") as file:
            json.dump(result, file, indent=4)
        logger.info(f"Successfully wrote {len(samples)} samples to {output_file}.")
    except Exception as e:
        logger.error(f"Failed to write samples to {output_file}: {e}")
        raise


def main(
    input_file,
    output_file,
    num_samples,
    model_name,
    temperature,
    temperature_answer,
    top_p,
    top_k,
):
    """Main function to handle the workflow."""
    logger.info("Starting the sample generation process.")
    logger.info(f"Loading {model_name}")
    try:
        generator = pipeline("text-generation", model=model_name)
        logger.info(f"Successfully loaded model {model_name}.")
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
        for entry in tqdm(cur_data):
            question = entry.get("question")
            gt_answer = entry.get("gt_answer")
            if not question:
                logger.warning(f"Skipping entry with missing question. Entry: {entry}")
                continue
            samples = generate_samples(
                generator, question, num_samples, temperature, top_p=top_p, top_k=top_k
            )
            answer = generate_samples(
                generator,
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
    )
