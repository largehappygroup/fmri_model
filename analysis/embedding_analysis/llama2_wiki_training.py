import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType

# Recommended for avoiding CUDA fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Model path
# model_name = "meta-llama/Llama-2-7b-hf"

# CodeBERT model path
model_name = "microsoft/codebert-base"

# CodeT5 model path small, base, large
# model_name = "Salesforce/codet5-small"
# model_name = "Salesforce/codet5-base"
# model_name = "Salesforce/codet5-large"

# Starcoder2 model path, 3B, 7B
# model_name = "bigcode/starcoder2-3b"
# model_name = "bigcode/starcoder2-7b"

# CodeGemma model path, 2B, 6B
# model_name = "Salesforce/codgemma-2b"
# model_name = "Salesforce/codgemma-6b"

# OpenAI embeddings model path
# model_name = "openai/text-embedding-3-small"
# model_name = "openai/text-embedding-3-large"


# Load in 4-bit using bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Disable caching for LoRA + checkpointing
model.config.use_cache = False

# Define LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # works for LLaMA2
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Load your dataset
dataset = load_dataset("wikitext", "wikitext-2-v1")

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./llama2_7b_lora_finetune_wiki",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_steps=500,
    save_total_limit=2,
    learning_rate=2e-4,
    fp16=True,  # Enable mixed precision
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer
)

# Train
trainer.train()

# Save LoRA adapter only (not full 7B weights)
model.save_pretrained("./llama2_7b_lora_adapter_wiki")
tokenizer.save_pretrained("./llama2_7b_lora_adapter_wiki")

print("finished")