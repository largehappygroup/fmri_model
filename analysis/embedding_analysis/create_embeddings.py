import os
# --- Setting up GPU ---
# os.environ["CUDA_VISIBLE_DEVICES"] = str(get_least_used_gpu())
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import csv
import torch
import pynvml
import pickle
import argparse
import numpy as np
import pandas as pd 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModel
)
from tqdm.auto import tqdm
from datasets import load_dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader

############################################################################
####### Setting Environment Variables & Input/Output Paths #################
############################################################################

# --- Checking if GPU is available ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
# --- Preparing dataset ---
csv_code_prompts = "/home/zachkaras/fmri/model/data/code_writing_prompts.csv" 
csv_prose_prompts = "/home/zachkaras/fmri/model/data/prose_writing_prompts.csv"

code_dataset = load_dataset("csv", data_files={"train": csv_code_prompts})
prose_dataset = load_dataset("csv", data_files={"train": csv_prose_prompts})

# Determines the max number of tokens to generate based on human participant responses
with open("midprocessing/max_response_lengths.pkl", 'rb') as f:
    max_response_lengths = pickle.load(f)

# --- Setting output ---
base_outpath = "/home/zachkaras/fmri/model/embeddings"
generated_code_outpath = f"{base_outpath}/code/generated_code.csv"
generated_prose_outpath = f"{base_outpath}/prose/generated_prose.csv"

with open(generated_code_outpath, 'a+') as f:
        cfile = csv.writer(f)
        cfile.writerow(['model', 'stim_id', 'run_num', 'generated_text'])

with open(generated_prose_outpath, 'a+') as f:
        cfile = csv.writer(f)
        cfile.writerow(['model', 'stim_id', 'run_num', 'generated_text'])

# Load in 4-bit using bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

model_names = {
    # "codebert": "microsoft/codebert-base",
    # "codet5_small": "Salesforce/codet5-small",
    # "codet5_base": "Salesforce/codet5-base",
    # "codet5_large": "Salesforce/codet5-large",
    # "openai_small": "openai/text-embedding-3-small",
    # "openai_large": "openai/text-embedding-3-large",
    "starcoder2_3b": "bigcode/starcoder2-3b",
    "starcoder2_7b": "bigcode/starcoder2-7b",
    "codegemma_2b" : "google/codegemma-2b",
    "codegemma_7b" : "google/codegemma-7b",
    "deepseek_2b"  : "deepseek-ai/deepseek-coder-1.3b-base",
    "deepseek_6b"  : "deepseek-ai/deepseek-coder-6.7b-base"
    
}

###########################################################################
##################### Functions ###########################################
###########################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Setting parameters for running models")
    parser.add_argument("--temp", type=float, required=False, help='Sets the temperature for prompting models. If unset, huggingface defaults are used')
    parser.add_argument("--num_samples", type=int, required=False, help='If increasing the temperature, indicates the number of iterations to run. In a way, "recruiting participants"')
    return parser.parse_args()

def tokenize_for_generation(examples):
        # Assuming the text to embed is in the 'text' column
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length", # Or 'longest' or 'do_not_pad' depending on your needs
            max_length=64, # Adjust max_length as appropriate for your data and model
            return_tensors="pt" # Return PyTorch tensors
        )
        # return inputs

def decide_model(model_name):
    print(model_name)
    # if "codet5" in model_name or "codebert" in model_name:
    #     return AutoModel.from_pretrained(model_name, trust_remote_code=True)
    # elif "starcoder2" in model_name or "codegemma" in model_name:
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, quantization_config=bnb_config)
    # elif "openai" in model_name:
    #     return AutoModel.from_pretrained(model_name, trust_remote_code=True)
    except:
        print(f"Model {model_name} not recognized for embeddings extraction.")
        return None
        # raise ValueError(f"Model {model_name} not recognized or not supported for embeddings extraction.")

def get_least_used_gpu():
    pynvml.nvmlInit()
    min_mem = float('inf')
    best_gpu = 0
    for i in range(torch.cuda.device_count()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print(f"GPU /{i}: {mem.used / (1024 ** 2):.2f} MB used")
        if mem.used < min_mem:
            min_mem = mem.used
            best_gpu = i
    pynvml.nvmlShutdown()
    # return best_gpu
    return 1

def generate_and_capture_all(model, temperature, tokenizer, prompt, device, max_new_tokens=64):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    
    # make 10 iterations for 10 runs
    # hidden_states = {}
    
    with torch.no_grad():
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            output_hidden_states = True,
            return_dict = True
        )
        
    print(vars(outputs))
    
    

def generate_and_capture(model, temperature, tokenizer, prompt, device, max_new_tokens=64):
    # Step 0: Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    generated_ids = input_ids.clone()
    hidden_states_per_step = []
    generated_tokens = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        # Get last hidden state from final layer for the most recent token
        last_token_hidden = outputs.hidden_states[-1][:, -1, :]  # shape: (1, hidden_dim)
        hidden_states_per_step.append(last_token_hidden.cpu())

        # Sample next token
        logits = outputs.logits[:, -1, :]
        
        # If collecting more than one "sample" from each model
        if temperature == None:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated_tokens.append(next_token.item())

        # Append new token
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

        # Optional: break early at EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode full generation
    full_generated_ids = generated_ids[0, input_ids.shape[-1]:]  # exclude prompt
    generated_text = tokenizer.decode(full_generated_ids, skip_special_tokens=True)

    # Stack hidden states (num_steps, hidden_dim)
    hidden_states_tensor = torch.cat(hidden_states_per_step, dim=0)  # (steps, dim)

    return generated_text, hidden_states_tensor


def generation_wrapper(dataset, task, model_name, temperature, num_samples):
    
    generated_outpath = generated_code_outpath if task == 'code' else generated_prose_outpath

    # --- Function for saving output ---
    def record_in_csv(question_num, generated_text, hidden_states_tensor, run=0):
        with open(generated_outpath, 'a+') as f:
            cfile = csv.writer(f)
            cfile.writerow([model_name, question_num, run, generated_text])
        
        # Saving embeddings
        outpath = f"embeddings/{task}/{model_name}/question_{question_num}_run_{run}.pt"
        torch.save(hidden_states_tensor, outpath)

    
    print(f"Generating and saving output for {task} task, question...")
    for i, row in enumerate(dataset['train']):
        print(i)
        prompt = row["text"]
        question_num = row["stim_id"]
        max_tokens = max_response_lengths[task][int(question_num)]
        
        if temperature == None:
            # generated_text,hidden_states_tensor = generate_and_capture(model, temperature, tokenizer, prompt,device, max_new_tokens=max_tokens)
            generated_text,hidden_states_tensor = generate_and_capture_all(model, temperature, tokenizer, prompt,device, max_new_tokens=max_tokens)
            record_in_csv(question_num, generated_text, hidden_states_tensor)
            
        else: # if temperature is set, collect a few samples for each prompt to get a variety of responses
            for i in range(num_samples):
                generated_text, hidden_states_tensor = generate_and_capture(model, temperature, tokenizer, prompt,device, max_new_tokens=max_tokens)
                record_in_csv(question_num, generated_text, hidden_states_tensor, run=i+1)
        
        
        # # --- Saving output ---
        # # Saving generated text
        # with open(generated_outpath, 'a+') as f:
        #     cfile = csv.writer(f)
        #     cfile.writerow([model_name, question_num, generated_text])
        
        # # Saving embeddings
        # outpath = f"embeddings/{task}/{model_name}/question_{question_num}.pt"
        # torch.save(hidden_states_tensor, outpath)


#############################################################################
############# Main Functionality for Creating Embeddings ####################
#############################################################################


args = parse_args()
temperature = args.temp # Could be none
num_samples = args.num_samples if args.num_samples != None else 5

for model_name, model_path in model_names.items():
    
    # redo check after finalizing output
    outpath_code = f"embeddings/code/{model_name}"
    outpath_prose = f"embeddings/prose/{model_name}"
    if not os.path.exists(outpath_code):
        os.system(f"mkdir {outpath_code}")
    # else:
    #     continue
        
    if not os.path.exists(outpath_prose):
        os.system(f"mkdir {outpath_prose}")
    
    # if os.path.exists(outpath_code) and os.path.exists(outpath_prose):
    #     print(f"Embeddings for {model_name} already exist. Skipping...")
    #     continue
   
    print(f"Loading tokenizer and model: {model_name}")
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
    model = decide_model(model_path)
    
    if model is None:
        continue

    model.to(device) # Move model to appropriate device (e.g., GPU if available)
    model.eval() # Set model to evaluation mode
    print(f"Model on {device}\nTokenizing dataset")
    
    # generating/saving text and embeddings based on prompts
    generation_wrapper(code_dataset, 'code', model_name, temperature, num_samples)
    # generation_wrapper(prose_dataset, 'prose', model_name, temperature, num_samples)

    # break

    print(f"Memory usage: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB out of {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
    
    # Garbage collection
    del model, tokenizer
    torch.cuda.empty_cache()  # Clear GPU memory

