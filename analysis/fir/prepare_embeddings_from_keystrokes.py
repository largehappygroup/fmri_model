import os
import re
import ast
import math
import pickle
import argparse
# import numpy as np
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


# Goal is to make discrete prompts for an LLM based on participants' keystrokes

# Need some way of concetanating keystrokes into meaningful chunks

# TODO use real characters for printable keys
# TODO use newline for enter \t for tab, etc.

'''
# should start with prompt, then add on successive characters and try processing the answer
def injest_vol_text(vol_text):
    patterns = [
        '[OPEN BRACKET]',
        '[CLOSE BRACKET]',
        '[CONTROL]',
        '[BACKSPACE]',
        '[ENTER]',
        '[SHIFT]',
        '[ESCAPE]',
        '[OPEN PARENTHESIS]', 
        '[CLOSE PARENTHESIS]', 
        '[SINGLE QUOTE]', 
        '[OPEN BRACE]', 
        '[OPEN BRACE]', 
        '[CLOSE BRACE]', 
        '[CLOSE BRACE]', 
        '[CLOSE BRACE]',
        '[TAB]',
        '[BACKSLASH]',  
        '[LEFT ARROW]',
        '[RIGHT ARROW]',
        '[UP ARROW]',
        '[DOWN ARROW]' 
    ]

    # s = "size[CLOSE BRACE]9"
    #s = "abcxxyz"
    i = 0
    results = []

    while i < len(vol_text):
        match = None
        for p in patterns:
            if vol_text.startswith(p, i):
                match = p
                break
        if match:
            results.append(match)
            i += len(match)
        else:
            results.append(vol_text[i])
            i += 1
    
    return results
'''

def fill_in_the_middle(text, prefix, suffix):
    
    # prefix should be the question text and the keystrokes so far
    # wrap in <PRE><SUF><MID>
    # each model has their own conventions, but I'll replace those at the time of prompting
    
    
    # suffix should be the remainder of current token ('ate' for isDuplic(ate) )
    
    
    
    # 
    
    pass


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

    return combined_text


def process_participant(task, person, participant_path):

    df_path = f"{participant_path}/{task}_keystrokes_by_volume.csv"
    try:
        vol_keystroke_df = pd.read_csv(df_path)
    except:
        print(f"No file for participant {person} for {task}")
        return
    
    # TODO need a greedy algorithm for ingesting characters

    # special = ['[CONTROL]', '[BACKSPACE]']
    answer = ''
    # prev_question = '[]'
    # #prev_token = ''
    for i,row in vol_keystroke_df.iterrows():
        question_num = row['question_num']
        # print(question_num, question_num == '[]')
       

        # if row['question_num'] != prev_question:
        #     answer += '\n\n'
        #     prev_question = row['question_num']
        #if question_num != '[4]' or question_num or '[]' or question_num or '[8]':
        #    break
        
        vol_text = ast.literal_eval(row['keystrokes'])
        
        if len(vol_text) == 0:
            continue
        
        # combining shift sequences
        shift_combined = combine_shift_sequences(vol_text)
        print(vol_text, shift_combined)
        
        # TODO - Fill in the middle
        formatted = fill_in_the_middle(question_num, text=shift_combined)

        # if isinstance(vol_text, float):
        #     continue
        # print(type(vol_text), vol_text)
        
        # text = injest_vol_text(vol_text)
        # # print(text, type(text))
        # #print("NEW ROW", text)
        
        # # could combine text across rows
        # # until we hit a special character?
        # prev_token = ''
        # for s in text:
        #     print(s)
        #     if s in special and s == prev_token:
        #         continue
        #     else:
        #         prev_token = s
        #     #print(s)
        #     answer += s
    # print(answer)
        # print(type(text), text)
        # len_prev = 0
        # for s in text:
            
        #     if s == '[BACKSPACE]':
        #         test = answer
        #         answer = answer[:-len_prev] # if s == '[BACKSPACE]' else answer
        #         print(f"len_prev: {len_prev} before: {test}, {s}, after: {answer}")
        #     else:
        #         len_prev = len(s)
        #         answer += s
        #     print(answer)
        # print(answer)


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
