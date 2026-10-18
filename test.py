import argparse
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate_text(
    prompt,
    model_dir="./gpt2-finetuned",
    max_new_tokens=80,
    temperature=0.8,
    top_p=0.95,
):
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"{model_dir} not found. Falling back to base GPT-2.")
        model_dir = "gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate text with GPT-2.")
    parser.add_argument("prompt", nargs="?", help="Text prompt")
    parser.add_argument("--model-dir", default="./gpt2-finetuned", help="Model folder")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Generated tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Creativity level")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling value")
    args = parser.parse_args()

    if args.prompt:
        print(
            generate_text(
                args.prompt,
                args.model_dir,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
            )
        )
        return

    print("Enter a prompt to generate text. Type 'exit' to stop.")
    while True:
        prompt = input("\nPrompt: ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue

        print(
            generate_text(
                prompt,
                args.model_dir,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
            )
        )


if __name__ == "__main__":
    main()
