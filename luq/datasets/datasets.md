## Formats
### Raw dataset
```json
{
    "data": [
        {
            "question": "Is this an example of a question?"‚
            "answer": "Yes, it is",
        },
        ...
    ]
}
```

### Sample dataset
```json
{
    "llm": llm_name,
    "raw_dataset": name of a raw dataset or a link to the raw dataset,
    "data": [
        {
            "question": "Is this a question?",
            "samples": ["Yes", "No", ...],
            "answer": "Yes",
            "samples_temp": 1,
            "answer_temp": 0.1,
            "top-p": xxx,
        },
        ...
    ]
}
````
