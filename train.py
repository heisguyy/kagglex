"""The script of training the model on the Image captioning dataset."""

import os

import torch
from datasets import load_dataset
from PIL import Image
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
)

ds = dataset = load_dataset("csv", data_files="data/metadata.csv")
train_ds = ds["train"]

MODEL_ID = "google/paligemma-3b-mix-224"
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)

DEVICE = "cuda"


def collate_fn(examples):
    texts = [f"caption {example['language']}" for example in examples]
    labels = [example["caption"] for example in examples]
    images = [
        Image.open(f"data/{example['file_name']}").convert("RGB")
        for example in examples
    ]
    tokens = processor(
        text=texts,
        images=images,
        suffix=labels,
        return_tensors="pt",
        padding="longest",
    )

    tokens = tokens.to(torch.bfloat16).to(DEVICE)
    return tokens


model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16
).to(DEVICE)

for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
    param.requires_grad = False

args = TrainingArguments(
    num_train_epochs=2,
    remove_unused_columns=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=100,
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    output_dir="kagglex-paligemma",
    bf16=True,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
)


trainer = Trainer(
    model=model, train_dataset=train_ds, data_collator=collate_fn, args=args
)

trainer.train(
    # resume_from_checkpoint="kagglex-paligemma/checkpoint-5000"
)

trainer.push_to_hub()
