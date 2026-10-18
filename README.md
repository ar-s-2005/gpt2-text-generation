
# GPT-2 Text Generation and Fine-Tuning

A Python project demonstrating how to fine-tune a pretrained GPT-2 language model on a small text dataset and generate text from custom prompts.

## Features

- Fine-tunes a pretrained GPT-2 model.
- Loads training text from `data.txt`.
- Generates text interactively using `generate.py`.
- Saves the fine-tuned model locally.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Make sure `data.txt` contains non-empty training examples, then run:

```bash
python train.py
```

## Generate Text

After training completes, run:

```bash
python generate.py
```

Enter a starting sentence when prompted. Type `exit` to quit.

## Limitations

This project fine-tunes a pretrained GPT-2 model; it does not train a language model from scratch. The current dataset is small, so generated output may be repetitive or inaccurate.

## Future Improvements

- Expand and improve the training dataset.
- Compare base GPT-2 and fine-tuned outputs.
- Add automated tests and training metrics.
