
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def main():
    print("Loading GPT-2 tokenizer and model...")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Loading dataset from data.txt...")

    dataset = load_dataset(
        "text",
        data_files={"train": "data.txt"},
    )

    # Remove blank text entries.
    dataset = dataset.filter(
        lambda example: bool(example["text"].strip())
    )

    if len(dataset["train"]) == 0:
        raise ValueError(
            "data.txt is empty. Add some non-empty training text."
        )

    # Tokenize the training text.
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )

    print("Tokenizing dataset...")

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        save_strategy="epoch",
        logging_steps=1,
        report_to="none",
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )

    print("Starting GPT-2 fine-tuning...")
    trainer.train()

    print("Saving fine-tuned model...")
   import os

os.makedirs("./gpt2-finetuned", exist_ok=True)

trainer.save_model("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

print("Model saved successfully!")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
