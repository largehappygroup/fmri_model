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


# TODO as input, I should take in the text that's been written so far, then the new text to be added
# def process_backspaces(text_pieces):
def process_backspaces(text_so_far, new_text):

    for token in new_text:
        one_match = re.search('<K:BS>', token)
        many_match = re.search(r'<K:BS x=([1-9]+)>', token)
        ctrl_match = re.search('<K:CTRL', token)
        
        # TODO - figure out why arrow keys are messing with output
        if re.search(r'(<K:L)|(<K:R)|(<K:U)|(<K:D)', token):
            print('ARROW KEY', token)
        
        if ctrl_match:
            continue
        
        if one_match:
            text_so_far = text_so_far[:-1]
        elif many_match:
            del_count = int(many_match.group(1))
            text_so_far = text_so_far[:-del_count]
        else:
            text_so_far += token

    return text_so_far


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

            if ch in separators:
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

    # return str.join('', combined_text)
    return combined_text



def get_question_text(task, question_num):
    question_df = code_question_df if task == 'code' else prose_question_df
    
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
    
    prev_question = -1
    accumulated_answer = ''

    for i,row in vol_keystroke_df.iterrows():
        curr_question = ast.literal_eval(row['question_num'])

        if curr_question != prev_question and curr_question != []:
            prev_question = curr_question
            question_num = curr_question[0]
            question_text = get_question_text(task, question_num)
            accumulated_answer = ''
            
        
        vol_text = ast.literal_eval(row['keystrokes'])
        
        if len(vol_text) == 0:
            continue
        
        # combining shift sequences
        shift_combined = combine_shift_sequences(vol_text)
        
        if shift_combined[-1] in separators:    
            next_text = ''
        else:
            next_text = find_next_sequence(i, vol_keystroke_df)
        
        # TODO - Fill in the middle
            # send in question number
            # text so far <PRE>
            # look ahead to find next keystrokes belonging to the same token <SUF>
            # current keystrokes to integrate <MID>
        
        formatted_text = fill_in_the_middle(str.join('', shift_combined), accumulated_answer, next_text)
        # print(formatted_text)
        
        integrated_text = process_backspaces(accumulated_answer, shift_combined)
        
        # print(f"CURRENT STRING: {accumulated_answer} OUTPUT: {processed_answer}")
        accumulated_answer = integrated_text
        
        # current text, prefix, suffix

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
