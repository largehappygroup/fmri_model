import os
import re
import ast
import math
import pickle
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Script to concatenate keystrokes into discrete chunks that are more interpretable.")
parser.add_argument("--computer", required=False, default='cumberland', help="This argument changes directory paths depending on whether I'm working on cumberland or my local computer")

args = parser.parse_args()

if args.computer == 'mymac':
    bass_path = "/Users/zacharykaras/Desktop"
elif args.computer == 'cumberland':
    bass_path = "/home/zachkaras/fmri"

with open(f"{bass_path}/fmri_model/midprocessing/shift_chars.pkl", 'rb') as f:
    shift_chars = pickle.load(f)
    
code_questions = "/home/zachkaras/fmri/fmri_model/midprocessing/code_writing_prompts.csv"
code_question_df = pd.read_csv(code_questions)

prose_question = "/home/zachkaras/fmri/fmri_model/midprocessing/prose_writing_prompts.csv"
prose_question_df = pd.read_csv(prose_question)

separators = [' ', '\r', '\t', '{', '}', ',', '[', ']', '(', ')', '+', '-', ';', '=', '<=', '>=', '==', '']


# Goal is to make discrete prompts for an LLM based on participants' keystrokes

# Need some way of concetanating keystrokes into meaningful chunks

# TODO use real characters for printable keys
# TODO use newline for enter \t for tab, etc.


def process_backspaces(text_pieces):
    updated_text = []
    for token in text_pieces:
        # print(updated_text)
        match = re.match(r'^<K:BS x=([1-9]+)>$', token)
        
        if re.search(r'<K:BS>', token):
            updated_text.pop()
        elif match:
            # print("matched")
            bs_count = int(match.group(1))
            # print("mutltiple backspaces", token, match.group(1))
            for i in range(bs_count, 2,-1):
                if len(updated_text) == 1: # if we're just left with the question_text, leave that as is
                    break                
                
                updated_text.pop()
                # print(f"popped: {i}, {updated_text}")
        else:
            updated_text.append(token)
    return updated_text
            # count = 

def fill_in_the_middle(text, prefix, suffix):
    
    formatted_text = f"<PRE>{prefix}<SUF>{suffix}<MID>{text}"
    # prefix should be the question text and the keystrokes so far
    # wrap in <PRE><SUF><MID>
    # each model has their own conventions, but I'll replace those at the time of prompting
    
    
    # suffix should be the remainder of current token ('ate' for isDuplic(ate) )
    return formatted_text
    

def find_next_sequence(i, keystroke_df):
    sequence = ''
    separators = [' ', '\r', '\t', '{', '}', ',', '[', ']', '(', ')', '+', '-', ';', '=', '<=', '>=', '==', '']
    
    tr = 0.8
    time_limit = 4 # 4 seconds
    volumes_ahead_limit = math.floor(time_limit/tr)
    vol_i = 0
    j = i + 1
    while j < len(keystroke_df):
        row_text = ast.literal_eval(keystroke_df.loc[j, 'keystrokes'])
        
        
        for ch in row_text:
            # print(ch)
            if ch in separators: # TODO - split by space, return, tab, braces/brackets, other punctuation
                return sequence
            else:
                sequence += ch
        
        j += 1
        vol_i += 1
        
        if vol_i == volumes_ahead_limit:
            return sequence

def combine_shift_sequences(vol_text):
    combined_text = []
    
    shifted = False
    for i,t in enumerate(vol_text):
        if shifted:
            shifted = False
            continue
        if re.search("<K:S", t):
            if i < (len(vol_text)-1):
                next_key = vol_text[i+1]
                try:
                    shifted_char = shift_chars[next_key]
                except:
                    print(f"No entry for {next_key}, {ascii(next_key)}")
                    continue
                combined_text.append(shifted_char)
                shifted = True
            else:
                combined_text.append(t)       
        else:
            combined_text.append(t)

    return str.join('', combined_text)



def get_question_text(task, question_num):
    question_df = code_question_df if task == 'code' else prose_question_df
    
    # print(type(question_df.loc[4, 'stim_id']), question_df.loc[4, 'stim_id'] )
    
    question_idx = np.where(question_df['stim_id'] == question_num)
    question_text = (list(question_df.loc[question_idx, 'text']))[0]
    
    return question_text


def process_participant(task, person, participant_path):

    df_path = f"{participant_path}/{task}_keystrokes_by_volume.csv"
    try:
        vol_keystroke_df = pd.read_csv(df_path)
    except:
        print(f"No file for participant {person} for {task}")
        return
    
    answer = []
    prev_question = -1
    question_parts = []
    question_text = ''
    # #prev_token = ''
    for i,row in vol_keystroke_df.iterrows():
        curr_question = ast.literal_eval(row['question_num'])

        if curr_question != prev_question and curr_question != []:
            # answer += '\n\n'
            # print("IF", i)
            prev_question = curr_question
            question_num = curr_question[0]
            question_text = get_question_text(task, question_num)
            question_parts = [question_text]
            
        
        vol_text = ast.literal_eval(row['keystrokes'])
        
        if len(vol_text) == 0:
            continue
        
        # combining shift sequences
        shift_combined = combine_shift_sequences(vol_text)
        
        if shift_combined[-1] not in separators:    
            next_text = find_next_sequence(i, vol_keystroke_df)
        else:
            next_text = ''
        
        # print(f"CURRENT: {shift_combined} NEXT: {next_text}")
        
        # TODO - Fill in the middle
            # send in question number
            # text so far <PRE>
            # look ahead to find next keystrokes belonging to the same token <SUF>
            # current keystrokes to integrate <MID>
        # prefix = ''
        # suffix = ''
        
        formatted = fill_in_the_middle(shift_combined, question_text, next_text)
        # print("FORMATTED", formatted)
        # print(type(shift_combined), shift_combined, question_text)
        # print("PARTS", question_parts)
        question_parts.append(shift_combined)
        question_parts = process_backspaces(question_parts)
        print(f"{str.join('', question_parts)}") # NEW TEXT: {shift_combined} THEN 

def main():

    datapath = f"{bass_path}/fmri_model/analysis/fir/midprocess"
    participants = os.listdir(datapath)

    for person in participants:
        print(person)
        participant_path = f"{datapath}/{person}"

        process_participant('code', person, participant_path)
        # process_participant('prose', person, participant_path)
        break

if __name__ == "__main__":
    main()
