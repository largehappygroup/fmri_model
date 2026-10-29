import os
# --- Setting up GPU ---
# os.environ["CUDA_VISIBLE_DEVICES"] = str(get_least_used_gpu())
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
import pynvml
import pickle
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel
)

############################################################################
####### Setting Environment Variables & Input/Output Paths #################
############################################################################

# --- Checking if GPU is available ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_names = {
    "starcoder2_3b": "bigcode/starcoder2-3b",
    "starcoder2_7b": "bigcode/starcoder2-7b",
    "codegemma_2b" : "google/codegemma-2b",
    "codegemma_7b" : "google/codegemma-7b",
    "deepseek_2b"  : "deepseek-ai/deepseek-coder-1.3b-base",
    "deepseek_6b"  : "deepseek-ai/deepseek-coder-6.7b-base",

    # General-purpose Decoder Only
    "gpt2"         : "openai-community/gpt2",
    "gpt2_xl"      : "openai-community/gpt2-xl",

    # Encoder Only
    "bert_base"    : "google-bert/bert-base-cased",
    "bert_large"   : "google-bert/bert-large-cased",
    "codebert"     : "microsoft/codebert-base",
}

# Encoder models are loaded with AutoModel and summarized by their [CLS]/<s> token
encoder_models = {"bert_base", "bert_large", "codebert"}

# Fill-in-the-middle tags (prefix, suffix, middle) for each code model family, taken from each tokenizer's special tokens
# DeepSeek's tags use full-width bars (U+FF5C) and U+2581 separators, not ASCII '|' and '_'
fim_tags = {
    "starcoder2": ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),
    "codegemma" : ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"),
    "deepseek"  : ("<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>"),
}

###########################################################################
##################### Functions ###########################################
###########################################################################

def decide_model(model_name, model_path):
    print(model_path)
    model_class = AutoModel if model_name in encoder_models else AutoModelForCausalLM
    try:
        return model_class.from_pretrained(model_path, 
                                           trust_remote_code=True, 
                                           torch_dtype=torch.bfloat16,
                                        #    quantization_config=quant_config,
                                           attn_implementation='eager')
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
        
def format_keystrokes(keystrokes, model_name):
    # keystrokes is {'prefix', 'suffix', 'middle'} from prepare_prompts_from_keystrokes.py
    prefix, suffix, middle = keystrokes['prefix'], keystrokes['suffix'], keystrokes['middle']
    family = model_name.rsplit('_', 1)[0]  # e.g. "starcoder2_3b" -> "starcoder2"
    if family in fim_tags:
        pre_tag, suf_tag, mid_tag = fim_tags[family]
        # middle goes after the middle tag, so the last token is the most recent keystroke
        return f"{pre_tag}{prefix}{suf_tag}{suffix}{mid_tag}{middle}"
    # gpt2 and encoders weren't trained with FIM, so give them the text in reading order
    return f"{prefix}{middle}{suffix}"

def generate_embeddings(keystrokes, model, tokenizer):

    # truncation_side is set to "left" in main(), so the most recent keystrokes are kept
    input_ids = tokenizer(keystrokes, 
                          truncation=True, 
                          max_length=tokenizer.model_max_length, 
                          return_tensors="pt").input_ids.to(device)
    # print(model.config._name_or_path, type(model), tokenizer.convert_ids_to_tokens(input_ids[0]))
    
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
        layer = hidden_state_layer.squeeze(0).float().cpu()
        hidden_states[f'layer_{i}'] = layer

    # save hidden states in directories 
    return hidden_states

def extract_summary_token(embedding, token_idx=-1):
    # -1 = last token (decoders), 0 = [CLS]/<s> token (encoders)
    sm_vector = embedding[token_idx]
    return sm_vector.numpy()

def pluck_intermediate_layers(embeddings, num_samples=8):
    all_layers = list(embeddings.keys())
    num_layers = len(all_layers)
    indices = np.linspace(0, num_layers - 1, num_samples, dtype=int)
    
    # intermediate_layer_idx = [n for n in range(0, num_layers, step)]
    intermediate_layer_labels = [all_layers[i] for i in indices]
    return intermediate_layer_labels

def z_score_embedding_dictionary(embeddings, layers):

    keys = list(embeddings.keys())
    for l in layers:
        X = torch.stack([torch.from_numpy(embeddings[k][l]) for k in keys])
        X = (X-X.mean(dim=0)) / X.std(dim=0).clamp_min(1e-8)
        for k,row in zip(keys, X):
            embeddings[k][l] = row.numpy()
    return embeddings

        
def process_participants(model_name, model, tokenizer, task):
    participant_path = "/tank/home/zachkaras/fmri_model/analysis/fir/midprocess"
    participants = os.listdir(participant_path)
    model_outpath = f"/tank/home/zachkaras/fmri_model_data/fir_embeddings_params/{model_name}"
    look_ahead_times = [0, 1, 3, 5, 10]
    summary_token_idx = 0 if model_name in encoder_models else -1
    
    if not os.path.isdir(model_outpath):
        os.mkdir(model_outpath)
    
    for person in participants:
        for t in look_ahead_times:
            print(f"{person}")
            keystroke_path = f"{participant_path}/{person}/{task}-look_ahead_by_{t}-formatted_keystrokes.pkl"
                
            try:
                with open(keystroke_path, 'rb') as f:
                    keystroke_dict = pickle.load(f)
            except:
                #print(f"No file for {person} on {task}")
                continue
            
            participant_outpath = f"{model_outpath}/{person}"
            if not os.path.exists(participant_outpath):
                os.mkdir(participant_outpath)

            keystroke_embeddings_by_vol = {}
            for vol, keystrokes in keystroke_dict.items():
                # volumes without keystrokes are stored as an empty dict
                if not keystrokes:
                    continue

                prompt = format_keystrokes(keystrokes, model_name)
                embeddings = generate_embeddings(prompt, model, tokenizer)
                layer_samples = pluck_intermediate_layers(embeddings)
                # keyed by volume, since the keystroke dicts can't be used as keys
                keystroke_embeddings_by_vol[prompt] = {layer : extract_summary_token(embeddings[layer], summary_token_idx) for layer in layer_samples}
                # summary_chosen_layers
            if not keystroke_embeddings_by_vol:
                print(f"No keystrokes for {person} on {task}, look_ahead_by_{t}")
                continue
            # z-score each feature across volumes, separately per layer
            z_scored_embeddings = z_score_embedding_dictionary(keystroke_embeddings_by_vol, layer_samples)
                
            with open(f"{participant_outpath}/{task}_look_ahead_by_{t}-keystroke_embeddings.pkl", 'wb') as f:
                pickle.dump(z_scored_embeddings, f)
                # break
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
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
        tokenizer.truncation_side = "left"
        model = decide_model(model_name, model_path)
        
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
