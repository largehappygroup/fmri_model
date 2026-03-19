import os
import re
import pickle
from collections import defaultdict
import json

best_models = ['code-deepseek_6b-ndelays_4-look_ahead_by_10',
               'code-codegemma_7b-ndelays_4-look_ahead_by_10',
               'code-codegemma_7b-ndelays_16-look_ahead_by_0',
               'prose-codegemma_7b-ndelays_10-look_ahead_by_5',
               'prose-deepseek_6b-ndelays_20-look_ahead_by_10']

participant_base_path = "/data/zachkaras/fmri_model_data/ridge_regression_pca_params"
participants = os.listdir(participant_base_path)

def nested_dict():
    return defaultdict(nested_dict)

def count_backspaces(keys):
    bs_pattern = r'<K:BS[^>]*>'
    backspaces = 0
    bs_instances = re.finditer(bs_pattern, keys)
    
    for k in bs_instances:
        numerous_pattern = r'=([0-9]+)+>'
        string = k.group()

        if re.search(numerous_pattern, string):
            num_backspaces = (re.search(numerous_pattern, string)).group(1)

            backspaces += int(num_backspaces)
        else:
            backspaces += 1

    return backspaces

def count_special_chars(keys):
    
    sum_special_chars = 0
    special_pattern = r'<K:(?!BS)[^>]+>'

    sc_instances = re.finditer(special_pattern, keys)
    for k in sc_instances:
        string = k.group()

        if re.search('[0-9]+>', string) and not re.search(r'^<K:S|^<K:CTRL', string):
            numerous_pattern = r'=([0-9]+)+>'

            if re.search(numerous_pattern, string):
                num_chars = (re.search(numerous_pattern, string)).group(1)

                sum_special_chars += int(num_chars)
        else:
            sum_special_chars += 1

    return sum_special_chars


pattern = r'<[^>]+>'
typing_counts = nested_dict()

for p in participants:
    for task in ['code', 'prose']:
        keystroke_path = f"/home/zachkaras/fmri_model/analysis/fir/midprocess/{p}/{task}-new_keystrokes.pkl"
        
        if not os.path.exists(keystroke_path):
            continue

        with open(keystroke_path, 'rb') as f:
            key_dictionary = pickle.load(f)

        total_keys_pressed = 0
        total_backspaces = 0
        for n,keys in key_dictionary.items():
            
            clean_keys = re.sub(pattern, '', keys) # number of keys without special chars
            backspaces = count_backspaces(keys) # number of backspaces
            special_chars = count_special_chars(keys) # number of special chars

            total_backspaces += backspaces 
            total_keys_pressed += len(clean_keys) + backspaces + special_chars

            
        typing_counts[p][task] = {'total_keys_pressed' : total_keys_pressed, 
                                  'total_backspaces'   : total_backspaces}

typing_counts = json.loads(json.dumps(typing_counts))

with open("typing_counts.pkl", 'wb') as f:
    pickle.dump(typing_counts, f)



