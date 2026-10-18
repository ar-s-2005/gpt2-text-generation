
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


MODEL_PATH = "./gpt2-finetuned"


def main():
    print("Loading your fine-tuned GPT-2 model...")

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

    model.eval()
    device = torch.device("cpu")
    model.to(device)

    print("Model loaded successfully!")
    print("Type 'exit' to quit.\n")

    while True:
        prompt = input("Enter a starting sentence: ").strip()

        if prompt.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not prompt:
            print("Please enter some text.\n")
            continue

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.8,
                top_p=0.92,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        )

        print("\nGenerated text:")
        print(generated_text)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
