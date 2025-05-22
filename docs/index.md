# Language Models Uncertainty Quantification (LUQ)

## Get Started

### Install LUQ:
```bash
pip install luq
```

### Use LUQ model for UQ
```py
import luq
from luq.models import MaxProbabilityEstimator

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
# Create text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# sample from LLM
samples = luq.llm.generate_n_samples_and_answer(
    pipeline,
    prompt="A, B, C, or D"
)

mp_estimator = MaxProbabilityEstimator()
print(mp_estimator.estimate_uncertainty(samples))
```

## Uncertainty Quantification Methods
Generally the uncertainty quantification in LUQ sample multiple responses from an LLM and analyse the

| Method  | Class in LUQ | Note | Reference     |
| ------------- | ------------- | ------------- | ------------- |
| Max Probability | `luq.models.max_probability` | Estimates uncertainty as one minus the probability of the most likely sequence in the list of samples. | -  |
| Top K Gap | `luq.models.top_k_gap` | Estimates uncertainty by measuring the gap between the most probable sequence and the k-th most probable one. | -  |
| Predictive Entropy  | `luq.models.predictive_entropy` | Uncertainty is estimated by computing the entropy of probabilities obtained from sampled sequences. | https://arxiv.org/pdf/2002.07650 |
| p(true)  | `luq.models.p_true` | Uncertainty is estimated by computing the entropy of probabilities obtained from sampled sequences. | https://arxiv.org/pdf/2002.07650 |
| Semantic Entropy  | `luq.models.semantic_entropy` | Uncertainty is estimated by performing semantic clustering of LLM responses and calculating the entropy across the clusters. | https://arxiv.org/abs/2302.09664 |
| Kernel Language Entropy  | `luq.models.kernel_language_entropy` | Uncertainty is estimated by performing semantic clustering of LLM responses and calculating the entropy across the clusters. | https://arxiv.org/abs/2405.20003 |

## Contributing
### Use pre-commit
```bash
pip install pre-commit
pre-commit install
```

## Pipeline for dataset creation
### Step 1. Create a processed version of a dataset.
```bash
mkdir data/coqa
python scripts/process_datasets.py \
    --dataset=coqa \
    --output=data/coqa/processed.json
```

```python
import json

data = json.load(open("data/coqa/processed.json", "r"))
new_data = {"train": data["train"][:2], "validation": data["validation"][:2]}
json.dump(new_data, open("data/coqa/processed_short.json", "w"))
```


### Step 2. Generate answers from LLMs and augment the dataset with the dataset.
```bash
python scripts/add_generations_to_dataset.py \
    --input-file=./data/coqa/processed_short.json\
    --output-file=./data/coqa/processed_gen_short.json\
```
### Step 3. Check accuracy of the answers given
```bash
python scripts/eval_accuracy.py \
    --input-file=data/coqa/processed_gen_short.json \
    --output-file=data/coqa/processed_gen_acc_short.json \
    --model-name=gpt2 \
    --model-type=huggingface
```
### Step 4. Upload the dataset to HuggingFace
```bash
python scripts/upload_dataset.py \
    --path=data/coqa/processed_gen_acc_short.json \
    --repo-id your-username/dataset-name \
    --token your-huggingface-token
```

## Datasets end-to-end
In order to generate a dataset:
```python
python scripts/gen_dataset.py --input-file=./data/dummy_data/raw_dummy.json --output-file=./output.json
```

When a dataset is created we can augment it with accuracies checked by another LLM:
```python
python scripts/eval_accuracy.py --input-file ./output.json --output-file test.json --model-name=gpt2 --model-type=huggingface
```

