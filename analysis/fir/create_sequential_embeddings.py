import os
# --- Setting up GPU ---
# os.environ["CUDA_VISIBLE_DEVICES"] = str(get_least_used_gpu())
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import csv
import math
import torch
import pynvml
import pickle
# import argparse
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
# csv_code_prompts = "/home/zachkaras/fmri/model/data/code_writing_prompts.csv" 
# csv_prose_prompts = "/home/zachkaras/fmri/model/data/prose_writing_prompts.csv"

# code_dataset = load_dataset("csv", data_files={"train": csv_code_prompts})
# prose_dataset = load_dataset("csv", data_files={"train": csv_prose_prompts})

# # Determines the max number of tokens to generate based on human participant responses
# with open("midprocessing/max_response_lengths.pkl", 'rb') as f:
#     max_response_lengths = pickle.load(f)

# # --- Setting output ---
# base_outpath = "/home/zachkaras/fmri/model/embeddings"
# generated_code_outpath = f"{base_outpath}/code/generated_code.csv"
# generated_prose_outpath = f"{base_outpath}/prose/generated_prose.csv"

# with open(generated_code_outpath, 'a+') as f:
#         cfile = csv.writer(f)
#         cfile.writerow(['model', 'stim_id', 'run_num', 'generated_text'])

# with open(generated_prose_outpath, 'a+') as f:
#         cfile = csv.writer(f)
#         cfile.writerow(['model', 'stim_id', 'run_num', 'generated_text'])

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

# TODO - iterate through the dictionaries and pass the prompts into the models
#      - save different layers along the way
#      - can delete parts that I don't need

# def parse_args():
#     parser = argparse.ArgumentParser(description="Setting parameters for running models")
#     parser.add_argument("--temp", type=float, required=False, help='Sets the temperature for prompting models. If unset, huggingface defaults are used')
#     parser.add_argument("--num_samples", type=int, required=False, help='If increasing the temperature, indicates the number of iterations to run. In a way, "recruiting participants"')
#     return parser.parse_args()

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
        return AutoModelForCausalLM.from_pretrained(model_name, 
                                                    trust_remote_code=True, 
                                                    quantization_config=bnb_config,
                                                    attn_implementation='eager')
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

# def generate_and_capture_all(model, tokenizer, prompt, device):
#     print(prompt)
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
#     print(len(input_ids[0]), input_ids)
#     return
#     attention_mask = torch.ones_like(input_ids)
    
#     hidden_states = {}

#     with torch.no_grad():
#         outputs = model(
#             input_ids = input_ids,
#             attention_mask = attention_mask,
#             output_hidden_states = True,
#             # output_attentions = True,
#             return_dict = True
#         )
    
#     for i, hidden_state_layer in enumerate(outputs.hidden_states):
#          layer = hidden_state_layer.squeeze(0).cpu()
#          hidden_states[f'layer_{i}'] = layer

#     # save hidden states in directories 
#     # print(len())
#     return hidden_states
#     # print(outputs.attentions)
    

# def generation_wrapper(dataset, task, model_name, temperature, num_samples):
    
#     generated_outpath = generated_code_outpath if task == 'code' else generated_prose_outpath

#     # --- Function for saving output ---
#     def record_in_csv(question_num, generated_text, hidden_states_tensor, run=0):
#         # with open(generated_outpath, 'a+') as f:
#         #     cfile = csv.writer(f)
#         #     cfile.writerow([model_name, question_num, run, generated_text])
        
#         # Saving embeddings
#         # outpath = f"embeddings/{task}/{model_name}/question_{question_num}_run_{run}.pt"
#         # outpath = f"embeddings/{task}/{model_name}/question_{question_num}_run_{run}.pt"
#         outpath = f"embeddings/{task}/{model_name}_question_{question_num}.pt"

#         torch.save(hidden_states_tensor, outpath)

    
#     print(f"Generating and saving output for {task} task, question...")
#     for i, row in enumerate(dataset['train']):
#         print(i)
#         prompt = row["text"]
#         question_num = row["stim_id"]
#         max_tokens = max_response_lengths[task][int(question_num)]
        
#         if temperature == None:
#             # generated_text,hidden_states_tensor = generate_and_capture(model, temperature, tokenizer, prompt,device, max_new_tokens=max_tokens)
#             hidden_states_tensor = generate_and_capture_all(model, tokenizer, prompt, device)
#             # record_in_csv(question_num, None, hidden_states_tensor)
            
#         else: # if temperature is set, collect a few samples for each prompt to get a variety of responses
#             for i in range(num_samples):
#                 hidden_states_tensor = generate_and_capture_all(model, tokenizer, prompt, device)
#                 # record_in_csv(question_num, None, hidden_states_tensor, run=i+1)
#         # break
        
        # --- Saving output ---
        # Saving generated text
        
        ### Not generating text with current approach
        
        # with open(generated_outpath, 'a+') as f:
        #     cfile = csv.writer(f)
        #     cfile.writerow([model_name, question_num, generated_text])
        
        # Saving embeddings
        # outpath = f"embeddings/{task}/{model_name}/question_{question_num}.pt"
        # torch.save(hidden_states_tensor, outpath)

'''
    all_layers = list(zscored_embeddings.keys())
    num_layers = len(all_layers)
    step = math.floor(num_layers/num_samples)
    
    intermediate_layer_idx = [n for n in range(0, num_layers, step)]
    intermediate_layer_labels = [all_layers[i] for i in intermediate_layer_idx]
    
    
    # I'd like to look at more than just the first layer, but I want to avoid just looking at every layer
    # I could do a sampling across the embeddings, maybe 5?
    # first layer, last layer, 3 intermediate layers
    
    # layered like an onion - aggregating the different layers for a given question
    onion = defaultdict()
    
    for layer_label in intermediate_layer_labels:
        this_layer = zscored_embeddings[layer_label]
        
        processed_layer = this_layer[0] if cls else process_layer(this_layer)
        # print(len(processed_layer))
        
        onion[layer_label] = processed_layer
        
    # Saving last layer
    last_layer_label = all_layers[-1]
    last_layer = zscored_embeddings[last_layer_label][0] if cls else process_layer(zscored_embeddings[last_layer_label])
'''        
        
def generate_embeddings(keystrokes, model, tokenizer):

    # TODO send current text into model
    input_ids = tokenizer(keystrokes, return_tensors="pt").input_ids.to(device)
    # print(len(input_ids[0]), input_ids)

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
    # print(len())
    return hidden_states


    # print(f"Generating and saving output for {task} task, question...")
    # for i, row in enumerate(dataset['train']):
    #     print(i)
    #     prompt = row["text"]
    #     question_num = row["stim_id"]
    #     max_tokens = max_response_lengths[task][int(question_num)]
        
    #     if temperature == None:
    #         # generated_text,hidden_states_tensor = generate_and_capture(model, temperature, tokenizer, prompt,device, max_new_tokens=max_tokens)
    #         hidden_states_tensor = generate_and_capture_all(model, tokenizer, prompt, device)
    # pass

def extract_cls(embedding):
    cls_vector = embedding[-1]
    return cls_vector

def pluck_intermediate_layers(embeddings, num_samples=4):
    num_layers = len(embeddings.keys())
    all_layers = list(embeddings.keys())
    num_layers = len(all_layers)
    step = math.floor(num_layers/num_samples)
    
    intermediate_layer_idx = [n for n in range(0, num_layers, step)]
    intermediate_layer_labels = [all_layers[i] for i in intermediate_layer_idx]
    return intermediate_layer_labels

def z_score_embedding_dictionary(embeddings):
    all_data = torch.stack(list(embeddings.values()), dim=2)
    mean = all_data.mean(dim=2)
    std = all_data.std(dim=2)
    return {k : (v-mean)/std for k,v in embeddings.items()}

        
def process_participants(model_name, model, tokenizer, task):
    participant_path = "/home/zachkaras/fmri/model/fir/data"
    participants = os.listdir(participant_path)
    model_outpath = f"/home/zachkaras/fmri/model/fir/midprocess/{model_name}"
    
    if not os.path.exists(model_outpath):
        os.mkdir(model_outpath)
    
    for person in participants:
        print(f"{person}")
        keystroke_path = f"{participant_path}/{person}/{task}_formatted_keystrokes.pkl"
            
        try:
            with open(keystroke_path, 'rb') as f:
                keystroke_dict = pickle.load(f)
        except:
            print(f"No file for {person} on {task}")
            continue
        
        participant_outpath = f"{model_outpath}/{person}"
        if not os.path.exists(participant_outpath):
            os.mkdir(participant_outpath)
        
        
        # TODO check to see if the keystrokes string already exists
        # if it does, refer back to those embeddings somehow...
        
        keystrokes_so_far = []
        for vol, keystrokes in keystroke_dict.items():
  
            if keystrokes == '':
                continue
            embeddings = generate_embeddings(keystrokes, model, tokenizer)
            zscored_emb = z_score_embedding_dictionary(embeddings)

            layer_samples = pluck_intermediate_layers(zscored_emb)
            print(layer_samples)

            # pull out embeddings layers at these keys
            # then I just want the CLS token embeddings
            cls_chosen_layers = {layer : extract_cls(zscored_emb[layer]) for layer in layer_samples}
            

            # print(hidden_states['layer_5'].shape)
            # test = (hidden_states['layer_5'])[-1]
            # layer_sample = []
            # test = (hidden_states['layer_9'])[-1]

            # outpath = "test9.pt"
            # torch.save(test, outpath)


            # print(test.shape)
            break
            # print(hidden_states.shape)
            
        break
    


#############################################################################
############# Main Functionality for Creating Embeddings ####################
#############################################################################

def main():
    # args = parse_args()
    # temperature = args.temp # Could be none
    # num_samples = args.num_samples if args.num_samples != None else 5

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
        
        # TODO iterate through participants
        # process code and prose
        
        process_participants(model_name, model, tokenizer, 'code')
        # process_participants(model_name, model_path, 'prose')
        break
        
        
        # for every participant, iterate through keystrokes dictionary to get associated keystrokes for each volume
        # if keystrokes are different from a previous set of keystrokes, generate embeddings
        # otherwise, use embeddings on record
        
        # redo check after finalizing output
        # outpath_code = f"embeddings/code/{model_name}"
        # outpath_prose = f"embeddings/prose/{model_name}"
        # if not os.path.exists(outpath_code):
        #     os.system(f"mkdir {outpath_code}")
        # # else:
        # #     continue
            
        # if not os.path.exists(outpath_prose):
        #     os.system(f"mkdir {outpath_prose}")
        
        # if os.path.exists(outpath_code) and os.path.exists(outpath_prose):
        #     print(f"Embeddings for {model_name} already exist. Skipping...")
        #     continue
    
        # # generation_wrapper(prose_dataset, 'prose', model_name, temperature, num_samples)

        # # break

        # print(f"Memory usage: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB out of {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
        
        # # Garbage collection
        # del model, tokenizer
        # torch.cuda.empty_cache()  # Clear GPU memory

if __name__=="__main__":
    main()