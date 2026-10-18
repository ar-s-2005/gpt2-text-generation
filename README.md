# GPT-2 Text Generation and Fine-Tuning

A Python project demonstrating how to fine-tune a pretrained GPT-2 language model for text generation using Hugging Face Transformers.

## Features
- Fine-tunes a pretrained GPT-2 model on a custom text dataset.
- Loads training data from `data.txt`.
- Generates text interactively using `generate.py`.
- Saves the fine-tuned model locally.
- Uses PyTorch and Hugging Face libraries.

## Technologies
- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train.py
```

## Generate Text

```bash
python generate.py
```

## Dataset

The training examples are stored in `data.txt`. The current dataset is small and intended for experimentation.

## Limitations

This project fine-tunes an existing GPT-2 model; it does not train a language model from scratch. The small dataset may result in repetitive or inaccurate generated text.

## Future Improvements
- Expand and improve the training dataset.
- Compare base GPT-2 and fine-tuned outputs.
- Add automated tests and training metrics.