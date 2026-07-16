import os
# --- Setting up GPU ---
# os.environ["CUDA_VISIBLE_DEVICES"] = str(get_least_used_gpu())
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# import csv
# import math
import torch
import pynvml
import pickle
# import argparse
import numpy as np
# import pandas as pd 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModel
)
# from tqdm.auto import tqdm
# from datasets import load_dataset
# import torch.nn.functional as F
# from torch.utils.data import DataLoader

############################################################################
####### Setting Environment Variables & Input/Output Paths #################
############################################################################

# --- Checking if GPU is available ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
# Load in 4-bit using bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# Commenting most of the models out for demonstration purposes
model_names = {
    # "starcoder2_3b": "bigcode/starcoder2-3b",
    # "starcoder2_7b": "bigcode/starcoder2-7b",
    # "codegemma_2b" : "google/codegemma-2b",
    # "codegemma_7b" : "google/codegemma-7b",
    # "deepseek_2b"  : "deepseek-ai/deepseek-coder-1.3b-base",
    "deepseek_6b"  : "deepseek-ai/deepseek-coder-6.7b-base"
}

###########################################################################
##################### Functions ###########################################
###########################################################################

# def tokenize_for_generation(examples):
#         # Assuming the text to embed is in the 'text' column
#         return tokenizer(
#             examples["text"],
#             truncation=True,
#             padding="max_length", # Or 'longest' or 'do_not_pad' depending on your needs
#             max_length=64, # Adjust max_length as appropriate for your data and model
#             return_tensors="pt" # Return PyTorch tensors
#         )
#         # return inputs

def decide_model(model_name):
    print(model_name)
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, 
                                                    trust_remote_code=True, 
                                                    quantization_config=bnb_config,
                                                    attn_implementation='eager')
    except:
        print(f"Model {model_name} not recognized for embeddings extraction.")
        return None

# def get_least_used_gpu():
#     pynvml.nvmlInit()
#     min_mem = float('inf')
#     best_gpu = 0
#     for i in range(torch.cuda.device_count()):
#         handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#         mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
#         if mem.used < min_mem:
#             min_mem = mem.used
#             best_gpu = i
#     pynvml.nvmlShutdown()
#     return 1      
        
def generate_embeddings(keystrokes, model, tokenizer):

    input_ids = tokenizer(keystrokes, return_tensors="pt").input_ids.to(device)

    attention_mask = torch.ones_like(input_ids)
    
    hidden_states = {}

    with torch.no_grad():
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            output_hidden_states = True,
            # output_attentions = True,
            return_dict = True
        )
    
    for i, hidden_state_layer in enumerate(outputs.hidden_states):
        layer = hidden_state_layer.squeeze(0).cpu()
        hidden_states[f'layer_{i}'] = layer

    # save hidden states in directories 
    return hidden_states

def extract_summary_token(embedding):
    sm_vector = embedding[-1]
    return sm_vector.numpy()

def pluck_intermediate_layers(embeddings, num_samples=8):
    num_layers = len(embeddings.keys())
    all_layers = list(embeddings.keys())
    num_layers = len(all_layers)
    indices = np.linspace(0, num_layers - 1, num_samples, dtype=int)
    
    intermediate_layer_labels = [all_layers[i] for i in indices]
    return intermediate_layer_labels

def z_score_embedding_dictionary(embeddings):
    all_data = torch.stack(list(embeddings.values()), dim=2)
    mean = all_data.mean(dim=2)
    std = all_data.std(dim=2)
    return {k : (v-mean)/std for k,v in embeddings.items()}

        
def process_participants(model_name, model, tokenizer, task):
    participant_path = "data"
    participants = os.listdir(participant_path)
    model_outpath = f"model_output/{model_name}"
    look_ahead_times = [0, 5, 10] # [0, 1, 3, 5, 10]
    
    if not os.path.exists(model_outpath):
        os.mkdir(model_outpath)
    
    for person in participants:
        for t in look_ahead_times:
            print(f"{person}")

            keystroke_path = f"{participant_path}/{person}/{task}-look_ahead_by_{t}-formatted_keystrokes.pkl"
                
            try:
                with open(keystroke_path, 'rb') as f:
                    keystroke_dict = pickle.load(f)
            except:
                continue
            
            participant_outpath = f"{model_outpath}/{person}"
            if not os.path.exists(participant_outpath):
                os.mkdir(participant_outpath)
            
            keystroke_embeddings_by_vol = {}
            for vol, keystrokes in keystroke_dict.items():
    
                if keystrokes == '':
                    continue
                embeddings = generate_embeddings(keystrokes, model, tokenizer)
                zscored_emb = z_score_embedding_dictionary(embeddings)

                # Extracting just 3 layers here for demonstration purposes
                layer_samples = pluck_intermediate_layers(zscored_emb, num_samples=3)

                # pull out embeddings layers at these keys
                # then I just want the summary token embeddings
                summary_chosen_layers = {layer : extract_summary_token(zscored_emb[layer]) for layer in layer_samples}
                keystroke_embeddings_by_vol[keystrokes] = summary_chosen_layers
                
            with open(f"{participant_outpath}/{task}_look_ahead_by_{t}-keystroke_embeddings.pkl", 'wb') as f:
                pickle.dump(keystroke_embeddings_by_vol, f)
        #         break
        #     break
        # break
    

#############################################################################
############# Main Functionality for Creating Embeddings ####################
#############################################################################

def main():

    for model_name, model_path in model_names.items():
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
        # generation_wrapper(code_dataset, 'code', model_name)
        # process code and prose
        
        process_participants(model_name, model, tokenizer, 'code')
        process_participants(model_name, model, tokenizer, 'prose')
        
        print(f"Memory usage: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB out of {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
        
        # Garbage collection
        del model, tokenizer
        torch.cuda.empty_cache()  # Clear GPU memory
        
        # break


if __name__=="__main__":
    main()
