import re
import os
import pickle
import pandas as pd

max_response_lengths = {
    'prose': {
        0 : 0,
        1 : 0,
        2 : 0,
        3 : 0,
        4 : 0, 
        5 : 0,
        6 : 0,
        7 : 0,
        8 : 0    
    },
    'code': {
        0 : 0,
        1 : 0,
        2 : 0,
        3 : 0,
        4 : 0, 
        5 : 0,
        6 : 0,
        7 : 0,
        8 : 0
    }
}

def join_letters(split_response):
    
    # where output will be generated
    new_list = []
    
    # checking for and accumulating consecutive characters
    prev_chrs = ''
    newline_check = 0
    
    # checking for and accumulating consecutive operators
    operators = ['<', '>', '=', '!']
    operator_chrs = ''
    operator_check = 0
    
    for el in split_response:
        
        # operations to detect and append escape characters like tab and return
        if newline_check and (el == 'n' or el == 't'):
            new_list.append(f"\{el}")
            newline_check = 0
            continue
        elif el == '\\':
            newline_check = 1
        
        # handling consecutive operators
        elif el in operators:
            if len(prev_chrs) > 0: # if there's a string already being concatenated, put joined chrs into output list and reset
                new_list.append(prev_chrs)
                prev_chrs = ''
            operator_chrs += el
            if operator_check: # if there was an operator previously, combine with new operator and put into output list
                if len(operator_chrs) > 0:
                    new_list.append(operator_chrs)
                    operator_chrs = ''
            else:
                operator_check = 1

        # if it's alphanumeric, start or keep combining
        elif el.isalnum():
            prev_chrs += el
            if operator_check:
                if len(operator_chrs) > 0:
                    new_list.append(operator_chrs)
                    operator_chrs = ''
                    operator_check = 0
        
        # if it's not alphanumeric, dump string into output list and reset
        else:
            if len(prev_chrs) > 0:
                new_list.append(prev_chrs)
                prev_chrs = ''
            if el == '' or el == ' ': # filter out empty strings and spaces
                continue
            else:
                new_list.append(el)
    return [len(new_list), new_list]

# regex function for finding number of words in participants' responses
def split_answer(row, condition):
    split_by = '' if condition == 'code' else ' '

    if pd.isna(row['answer']):
        return 0
    
    split_response = re.split(split_by, row['answer'])
    
    if condition == 'code':
        response_len,rejoined_response = join_letters(split_response)
    if condition == 'prose':
        filtered_list = [el for el in split_response if el not in ['', '.', ' ']] # filter out periods and empty strings
        response_len = len(filtered_list)
        
    question_num = row['stimulus-id']
    
    if response_len > max_response_lengths[condition][question_num]:
        max_response_lengths[condition][question_num] = response_len

# regex function for finding number of tokens in code answers
def count_tokens(response_path, condition):
    try:
        df = pd.read_csv(response_path)
    except:
        print(f"file doesn't exist: {response_path}")
        return 0

    df.apply(lambda row : split_answer(row, condition), axis=1)

# directory path
participant_rootpath = "/home/zachkaras/fmri/fmri_model/data"
participants = os.listdir(participant_rootpath)

# iterate through files
for p in participants:
    
    # find processed-answers-3 files
    prose_answer_path = f"{participant_rootpath}/{p}/processed-answers-{p}-1.txt"
    code_answer_path = f"{participant_rootpath}/{p}/processed-answers-{p}-3.txt"
    
    word_count = count_tokens(prose_answer_path, condition='prose')
    token_count = count_tokens(code_answer_path, condition='code')

with open("/home/zachkaras/fmri/fmri_model/midprocessing/max_response_lengths.pkl", "wb") as f:
    pickle.dump(max_response_lengths, f)