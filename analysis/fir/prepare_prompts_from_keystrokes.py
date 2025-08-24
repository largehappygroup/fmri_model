import os
import re
import ast
import math
import pickle
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

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


class Text:
    """
    Editable multi-line text with a single cursor.
    - lines: internal list of list[str] (per-line characters)
    - line: current line index (1-based externally, 0-based internally)
    - col: cursor index within line (0-based)
    """
    def __init__(self, initial_text: str = ""):
        # raw_lines = initial_text.split('\r') if initial_text else [""]
        # self.lines: List[List[str]] = [list(l) for l in raw_lines]
        self.text: dict = { 0 : ''}
        self.line: int = 0  # 1-based
        # self.lines: List[List[str]] = [list(l) for l in raw_lines]
        # self.col: int = len(self.lines[0]) if self.lines else 0
        self.col: int = 0
        self.shifted: bool = False
        self.total_lines: int = 1
        self.question_text: str = ''
        # self.goal_col: Optional[int] = None  # For Up/Down column memory

    @property
    def total_lines_count(self) -> int:
        return len(self.lines)

    @property
    def line_number(self) -> int:
        return self.line  # 1-based

    @property
    def cursor_index(self) -> int:
        return self.col  # 0-based
    
    # @property
    # def shifted(self) -> bool:
    #     return self.shifted
    pass


def shift_line_numbers(answer):
    line = answer.line
    for i in range(len(answer.text.keys())-1, answer.line, -1):
        answer.text[i+1] = answer.text[i]
    # for i in range(answer.line, len(answer.text.keys())):
    #     temp = answer.text[i+1]
    #     answer.text[i+1] = answer.text[i]
        
        

    #     temp = answer.text[i]
        print(f"shifting line number {i} down to {i+1}")


# TODO as input, I should take in the text that's been written so far, then the new text to be added
# def process_backspaces(text_pieces):
def update_text_and_cursor_position(text_so_far, new_text, total_lines, line_number, cursor_index, shifted, processed_answer):

    for i,token in enumerate(new_text):
        one_match   = re.search('<K:BS>', token)
        many_match  = re.search(r'<K:BS x=([0-9]+)>', token)
        ctrl_match  = re.search('<K:CTRL', token)
        shift_match = re.search('<K:S', token)
        
        left_match  = re.search(r'(<K:L>)|(<K:L x=([0-9]+))', token)
        right_match = re.search(r'(<K:R>)|(<K:R x=([0-9]+))', token)
        up_match    = re.search(r'(<K:U>)|(<K:U x=([0-9]+))', token)
        down_match  = re.search(r'(<K:D>)|(<K:D x=([0-9]+))', token)
        
        # TODO - set up dictionary for line_number and text
        
        if re.search(r'\r', token):
            total_lines += 1
            line_number += 1
            processed_answer.line += 1
            processed_answer.total_lines += 1
            # TODO shift all the keys, values down below current line
            shift_line_numbers(processed_answer)
            processed_answer.text[processed_answer.line] = ''
            continue
            
            # any text to the right should also go onto the next line
            
            print("RETURN", token)
            
        if left_match:
            left_shift  = 1 if not left_match.group(3) else int(left_match.group(3))
            # continue
            # for left shifts, I need to pop characters off the end of the string and saved the popped characters
            # they can be the left and right strings
            # If I'm at the 1st index (as far left as I go), I need to go to the end of the previous line (decrement line number)
            print(f"LEFT: ", token, left_shift)
            
        elif re.search(r'<K:R', token):
            # continue
            right_shift = 1 if not right_match.group(3) else int(right_match.group(3))
            new_col_num = processed_answer.col + right_shift
            curr_line_length = len(processed_answer.text[processed_answer.line]) - 1
            
            # if cursor is at the end of the current line
            if new_col_num > curr_line_length:
                if processed_answer.line == (processed_answer.total_lines) - 1:
                    processed_answer.col = curr_line_length
                # if there are no other lines below, do nothing
                else:
                    right_shift_remainder = curr_line_length - processed_answer.col
                    processed_answer.line += 1
                    processed_answer.col = right_shift_remainder
                # go to next line, and change col to 0
            else:
                processed_answer.col = new_col_num
            
            # If I'm as far right as I can go, then I should stop
            # otherwise, increment character index and shift characters from right string to left string
            print(f"RIGHT: ", token, right_shift)
            continue
            
        elif re.search(r'<K:U', token):
            up_shift = 1 if not up_match.group(3) else int(up_match.group(3))
            new_line_num = processed_answer.line - up_shift 
            
            if new_line_num < 0:
                processed_answer.line = 0
            else:
                processed_answer.line = new_line_num
            # continue
            # If I'm at the top of the file, don't do anything
            # else, decrement the line_number and change line of text - maybe I should use a dictionary to represent text?
            
            print(f"UP: ", token, up_shift, processed_answer.line, new_line_num, processed_answer.total_lines, processed_answer.text)
            
        elif re.search(r'<K:D', token):
            down_shift  = 1 if not down_match.group(3) else int(down_match.group(3))
            new_line_num = processed_answer.line + down_shift
            
            if new_line_num > (processed_answer.total_lines) - 1:
                processed_answer.line = (processed_answer.total_lines) - 1
            else:
                processed_answer.line = new_line_num
            # continue
            # If I'm at the bottom of the file, do nothing
            # else, increment line number and change line of text to corresponding line
            
            print(f"DOWN: ", token, down_shift)
        
        if ctrl_match or shift_match:
            # if ctrl_match:
            if i == len(new_text)-1 and shift_match: # if the last token is a shift key
                shifted = True
                processed_answer.shifted = True
            continue
        
        # TODO check if I go onto previous line
        if one_match:
            text_so_far = text_so_far[:-1]
            processed_answer.text[processed_answer.line] = (processed_answer.text[processed_answer.line])[:-1]
            processed_answer.col -= 1
            
        elif many_match:
            del_count = int(many_match.group(1))
            text_so_far = text_so_far[:-del_count]
            processed_answer.text[processed_answer.line] = (processed_answer.text[processed_answer.line])[:-del_count]
            processed_answer.col -= del_count
        
        elif shifted:
            try:
                shifted_token = shift_chars[token]
            except:
                # print(f"Token {token} cannot be shifted")
                text_so_far += token
                processed_answer.text[processed_answer.line] += token
                
                processed_answer.col += len(token)
                
                continue
            text_so_far += shifted_token
            processed_answer.text[processed_answer.line] += token
            processed_answer.col += len(shifted_token)
            
            
            shifted = False
            processed_answer.shifted = False
        else:
            text_so_far += token
            processed_answer.text[processed_answer.line] += token
            
            processed_answer.col += len(token)

    return text_so_far, line_number, cursor_index, shifted


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

def process_keystrokes(vol_keystroke_df, task):
    
    prev_question = -1
    
    # TODO - I probably need a left and right answer to keep track of cursor
    uptodate_answer = ''
    line_number = 1 # 1-indexed
    cursor_index = 0
    total_lines = 1 # 1-indexed
    shifted = False
    
    # TODO - uptodate answer should be a dictionary where keys are line numbers and values are lines of text
    # I might need the values to be two lists
    processed_answer = Text()
    # print("OBJECT", processed_answer.col, processed_answer.shifted)
    
    
    for i,row in vol_keystroke_df.iterrows():
        curr_question = ast.literal_eval(row['question_num'])

        # line number and cursor_index should reset with each question
        if curr_question != prev_question and curr_question != []:
            prev_question = curr_question
            question_num = curr_question[0]
            question_text = get_question_text(task, question_num)
            processed_answer.question_text = get_question_text(task, question_num)
            
            uptodate_answer = ''
            line_number = 0
            cursor_index = 0
            
            # processed_answer = {}
            processed_answer = Text()
            print("OBJECT", processed_answer.col, processed_answer.shifted)
        
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
        prefix = f"{question_text} {uptodate_answer}"
        formatted_text = fill_in_the_middle(str.join('', shift_combined), prefix, next_text)
        # print(formatted_text)
        
        updated_text, new_line, new_cursor, shifted = update_text_and_cursor_position(uptodate_answer, shift_combined, total_lines, line_number, cursor_index, shifted, processed_answer)
        
        
        print(f"OBJECT UPDATE Total - lines: {processed_answer.total_lines}, row: {processed_answer.line}, col: {processed_answer.col}, bool: {processed_answer.shifted}, text: {processed_answer.text}")
        line_number = new_line
        cursor_index = new_cursor
        # print(updated_text)
        # print(f"CURRENT STRING: {accumulated_answer} OUTPUT: {processed_answer}")
        uptodate_answer = updated_text


def process_participant(task, person, participant_path):

    df_path = f"{participant_path}/{task}_keystrokes_by_volume.csv"
    try:
        vol_keystroke_df = pd.read_csv(df_path)
    except:
        print(f"No file for participant {person} for {task}")
        return
    
    process_keystrokes(vol_keystroke_df, task)
        
    # current text, prefix, suffix

def main():

    datapath = f"{bass_path}/fmri_model/analysis/fir/midprocess"
    participants = os.listdir(datapath)

    for person in participants:
        print(person)
        participant_path = f"{datapath}/{person}"

        process_participant('code', person, participant_path)
        # process_participant('prose', person, participant_path)
        # break

if __name__ == "__main__":
    main()
