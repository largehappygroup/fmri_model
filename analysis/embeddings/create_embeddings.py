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
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

# Move model to appropriate device (e.g., GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # Set model to evaluation mode

from datasets import load_dataset
import pandas as pd # You might need pandas for initial CSV loading/inspection

# --- Load your CSV dataset ---
# Option 1: Using datasets library (recommended)
# Assuming your CSV has a column named 'text' with the input data
csv_file_path = "your_dataset.csv" # <--- IMPORTANT: Change this to your CSV file path
dataset = load_dataset("csv", data_files={"train": csv_file_path}) # Or 'data_files=csv_file_path' if no splits

# If your text column has a different name, rename it or adjust tokenize_function
# Example if your column is named 'sentence':
# dataset = dataset.rename_column("sentence", "text")


# Tokenize the dataset
def tokenize_function_for_embeddings(examples):
    # Assuming the text to embed is in the 'text' column
    inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length", # Or 'longest' or 'do_not_pad' depending on your needs
        max_length=64, # Adjust max_length as appropriate for your data and model
        return_tensors="pt" # Return PyTorch tensors
    )
    return inputs

tokenized_dataset = dataset.map(tokenize_function_for_embeddings, batched=True)
# Remove original text column if you no longer need it to save memory
tokenized_dataset = tokenized_dataset.remove_columns([col for col in tokenized_dataset['train'].column_names if col != 'input_ids' and col != 'attention_mask'])

all_embeddings = []

# Prepare for iteration
# You might want to use a DataLoader for larger datasets to manage batches
from torch.utils.data import DataLoader

# Create a simple dataset class if needed, or use Hugging Face's built-in format
# For a simple case, you can directly iterate:
train_dataset = tokenized_dataset["train"]

# It's good practice to wrap your iteration in tqdm for progress tracking
from tqdm.auto import tqdm

batch_size = 8 # Adjust based on your GPU memory and desired speed

for i in tqdm(range(0, len(train_dataset), batch_size), desc="Extracting Embeddings"):
    batch_inputs = {k: torch.tensor([ex[k] for ex in train_dataset[i:i+batch_size]]) for k in train_dataset[0] if k in ['input_ids', 'attention_mask']}
    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

    with torch.no_grad(): # Important: Do not calculate gradients during inference
        outputs = model(**batch_inputs, output_hidden_states=True)

    # Get the embeddings from the last layer
    # For a causal LM, the last hidden state before the LM head is usually desired.
    # The shape is (batch_size, sequence_length, hidden_size)
    last_hidden_states = outputs.hidden_states[-1]

    # You might want to get the embedding for a specific token (e.g., CLS/SEP if applicable, or average)
    # For a simple sentence embedding, you can average across tokens:
    sentence_embeddings = torch.mean(last_hidden_states, dim=1) # Average over sequence length

    all_embeddings.append(sentence_embeddings.cpu()) # Move to CPU and store

# Concatenate all embeddings
all_embeddings = torch.cat(all_embeddings, dim=0)

# --- Save your embeddings ---
# You can save them as a PyTorch tensor, numpy array, or to a file (e.g., .npy, .csv)
torch.save(all_embeddings, "your_embeddings.pt")
# Or convert to numpy and save as .npy
import numpy as np
np.save("your_embeddings.npy", all_embeddings.numpy())

print(f"Finished extracting {all_embeddings.shape[0]} embeddings of size {all_embeddings.shape[1]}")