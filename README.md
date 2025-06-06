<div align="center">
<img src="docs/static/luq-logo.png">
</div>

<h3 align="center">
Language Models Uncertainty Quantification (LUQ)
</h3>

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ThUAboQQYgM5kJ0dCtwozSkC6WzW0GdE?usp=drive_link)
[![Pypi version](https://img.shields.io/pypi/v/luq)](https://pypi.org/project/luq/)
[![unit-tests](https://github.com/AlexanderVNikitin/luq/actions/workflows/test.yml/badge.svg?event=push)](https://github.com/AlexanderVNikitin/luq/actions?query=workflow%3ATests+branch%3Amain)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![codecov](https://codecov.io/gh/AlexanderVNikitin/luq/graph/badge.svg?token=ORX9NHH5ZU)](https://codecov.io/gh/AlexanderVNikitin/luq)

## Get Started

### Install LUQ:
```bash
pip install luq
```

### Use LUQ model for UQ
```python
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

## Tutorials
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexandervnikitin/luq/blob/main/tutorials/getting_started.ipynb) Introductory Tutorial [Getting started with LUQ](https://github.com/AlexanderVNikitin/luq/tutorials/getting_started.ipynb)  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexandervnikitin/luq/blob/main/tutorials/luq_datasets.ipynb) Working with [LUQ Datasets](https://github.com/AlexanderVNikitin/luq/tutorials/luq_datasets.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexandervnikitin/luq/blob/main/tutorials/predictive_entropy.ipynb) Using [Predictive Entropy](https://github.com/AlexanderVNikitin/luq/tutorials/predictive_entropy.ipynb)


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

## Contributing
We appreciate all contributions. To learn more, please check [CONTRIBUTING.md](CONTRIBUTING.md).

Install from sources:
```bash
git clone github.com/AlexanderVNikitin/luq
cd luq
pip install -e .
```

Run tests:
```bash
python -m pytest
```

## License
[MIT](LICENSE)