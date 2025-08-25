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
        self.line: int = 0 
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


def shift_line_numbers_down(answer):
    # line = answer.line
    print("SHIFTING ", len(answer.text.keys()), answer.line)
    for i in range(len(answer.text.keys())-1, answer.line, -1):
        print(i)
        answer.text[i+1] = answer.text[i]
        print(f"shifting line number {i} down to {i+1}")

def shift_line_numbers_up(answer):
    line = answer.line
    print("SHIFTING ", len(answer.text.keys()), answer.line)
    for i in range(answer.line, len(answer.text.keys())-1):
        answer.text[i] = answer.text[i+1]
    
    # for i in range(len(answer.text.keys())-1, answer.line, -1):
    
    #     print(i)
    #     answer.text[i+1] = answer.text[i]
    #     print(f"shifting line number {i} down to {i+1}")

def update_text_and_cursor_position(text_so_far, new_text, total_lines, line_number, cursor_index, shifted, answer):

    for i,token in enumerate(new_text):
        one_match   = re.search('<K:BS>', token)
        many_match  = re.search(r'<K:BS x=([0-9]+)>', token)
        ctrl_match  = re.search('<K:CTRL', token)
        shift_match = re.search('<K:S', token)
        
        left_match  = re.search(r'(<K:L>)|(<K:L x=([0-9]+))', token)
        right_match = re.search(r'(<K:R>)|(<K:R x=([0-9]+))', token)
        up_match    = re.search(r'(<K:U>)|(<K:U x=([0-9]+))', token)
        down_match  = re.search(r'(<K:D>)|(<K:D x=([0-9]+))', token)
        curr_line_length = len(answer.text[answer.line])
        # TODO - set up dictionary for line_number and text
        
        if re.search(r'\t', token):
            answer.col += 4
            answer.text[answer.line] += '    '
            continue
        
        if re.search(r'\r', token):
            # print("RETURN", token)
            total_lines += 1
            line_number += 1
            shift_line_numbers_down(answer)
            answer.line += 1
            answer.col = 0
            answer.total_lines += 1
            # TODO shift all the keys, values down below current line
            answer.text[answer.line] = ''
            continue
            
            # any text to the right should also go onto the next line
            
            
        if left_match:
            left_shift  = 1 if not left_match.group(3) else int(left_match.group(3))
            new_col_num = answer.col - left_shift
            if new_col_num < 0:
                if answer.line == 0:
                    answer.col = 0
                else:
                    left_shift_remainder = left_shift - answer.col
                    answer.line -= 1
                    answer.col = len(answer.text[answer.line]) - left_shift_remainder
            else:
                answer.col = new_col_num
            # continue
            # for left shifts, I need to pop characters off the end of the string and saved the popped characters
            # they can be the left and right strings
            # If I'm at the 1st index (as far left as I go), I need to go to the end of the previous line (decrement line number)
            # print(f"LEFT: ", token, left_shift)
            continue
            
        elif right_match:
            # continue
            right_shift = 1 if not right_match.group(3) else int(right_match.group(3))
            new_col_num = answer.col + right_shift
            curr_line_length = len(answer.text[answer.line])
            
            # if cursor is at the end of the current line
            if new_col_num > curr_line_length:
                if answer.line == (answer.total_lines) - 1:
                    answer.col = curr_line_length
                # if there are no other lines below, do nothing
                else:
                    right_shift_remainder = curr_line_length - answer.col
                    answer.line += 1
                    answer.col = right_shift_remainder
                # go to next line, and change col to 0
            else:
                answer.col = new_col_num
            
            # If I'm as far right as I can go, then I should stop
            # otherwise, increment character index and shift characters from right string to left string
            # print(f"RIGHT: ", token, right_shift)
            continue
            
        elif up_match:
            up_shift = 1 if not up_match.group(3) else int(up_match.group(3))
            new_line_num = answer.line - up_shift 
            
            if new_line_num < 0:
                answer.line = 0
            else:
                answer.line = new_line_num
            # continue
            # If I'm at the top of the file, don't do anything
            # else, decrement the line_number and change line of text - maybe I should use a dictionary to represent text?
            
            # print(f"UP: ", token, up_shift, answer.line, new_line_num, answer.total_lines, answer.text)
            continue
            
        elif down_match:
            down_shift  = 1 if not down_match.group(3) else int(down_match.group(3))
            new_line_num = answer.line + down_shift
            
            if new_line_num > (answer.total_lines) - 1:
                answer.line = (answer.total_lines) - 1
            else:
                answer.line = new_line_num
            # If I'm at the bottom of the file, do nothing
            # else, increment line number and change line of text to corresponding line
            
            # print(f"DOWN: ", token, down_shift, new_line_num, answer.total_lines)
            continue
        
        if ctrl_match or shift_match:
            # if ctrl_match:
            if i == len(new_text)-1 and shift_match: # if the last token is a shift key
                shifted = True
                answer.shifted = True
            continue
        
        # TODO check if I go onto previous line
        # print(shifted, answer.shifted)
        if one_match:
            text_so_far = text_so_far[:-1]
            left_string = (answer.text[answer.line])[:answer.col-1]
            right_string = (answer.text[answer.line])[answer.col:]
            # answer.text[answer.line] = (answer.text[answer.line])[:-1]
            answer.text[answer.line] = left_string + right_string
            
            if answer.col == 0:
                if curr_line_length == 0:
                    # put cursor at end of line above
                    # print(f"ZERO: curr_line_length {curr_line_length}, col: {answer.col}, total_lines: {answer.total_lines}, curr_line: {answer.line}, text: {answer.text}")
                    prev_line_length = 0 if answer.line == 0 else len(answer.text[answer.line-1])
                    if answer.line == 0:
                        combined_text = answer.text[answer.line]
                        answer.text[0] = combined_text
                    else:
                        combined_text = answer.text[answer.line-1] + answer.text[answer.line]
                        answer.text[answer.line-1] = combined_text
                    answer.col = prev_line_length - 1 if prev_line_length > 0 else 0
                    answer.total_lines = max(1, answer.total_lines - 1)
                    # answer.total_lines -= 1 if answer.total_lines > 1 else 1
                    answer.line -= 1 if answer.line > 0 else 0
                    # change column to length of previous line - 1
                    # decrement total lines
                    # decrement current line
                    # print(f"ZERO AFTER: curr_line_length {curr_line_length}, col: {answer.col}, total: {answer.total_lines}, curr line: {answer.line}, text: {answer.text}, combined: {combined_text}")
                # else:
                # curr_line_lengthx
                # if answer.text[answer.line]
                # pass
            else:
                answer.col -= 1
            
        elif many_match:
            del_count = int(many_match.group(1))
            while del_count > 0:
                if answer.col == 0:
                    if answer.line == 0:
                        break
                    else:
                        if len(answer.text[answer.line]) == 0:
                            shift_line_numbers_up(answer)
                        answer.line -= 1
                        answer.col = len(answer.text[answer.line])-1
                new_left_string = (answer.text[answer.line])[:answer.col-1]
                right_string = (answer.text[answer.line])[answer.col:]
                answer.text[answer.line] = new_left_string + right_string
                del_count -= 1
                answer.col -= 1
            
            text_so_far = text_so_far[:-del_count]
            '''
            if del_count > curr_line_length:
                del_remainder = del_count 
            
            mini_text = (answer.text[answer.line])[:answer.col]
            mini_text = mini_text[:-del_count]
            answer.text[answer.line] = (answer.text[answer.line])[:-del_count]
            if answer.col == 0:
                print(f"ZERO MANY: curr_line_length {curr_line_length}")
                pass
            answer.col -= del_count
            '''
       
        elif shifted:
            # print(shifted, shift_chars['9'], type(token), token)
            try:
                shifted_token = shift_chars[token]
            except:
                # print(f"Token {token} cannot be shifted")
                text_so_far += token
                answer.text[answer.line] += token
                
                answer.col += len(token)
                
                continue
            text_so_far += shifted_token
            answer.text[answer.line] += shifted_token
            answer.col += len(shifted_token)
            # print(f"newly shifted: {answer.text[answer.line]} and {shifted_token}")
            
            shifted = False
            answer.shifted = False
        else:
            text_so_far += token
            # answer.text[answer.line] += token
            left_string = (answer.text[answer.line])[:answer.col] + token
            right_string = (answer.text[answer.line])[answer.col:]
            new_string = left_string + right_string
            answer.text[answer.line] = new_string
            
            answer.col += len(token)

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
    question_text = ''
    
    answer = Text()
    # print("OBJECT", answer.col, answer.shifted)
    
    
    for i,row in vol_keystroke_df.iterrows():
        curr_question = ast.literal_eval(row['question_num'])
        # print(f"CURRENT QUESTION: ", curr_question)

        # line number and cursor_index should reset with each question
        if curr_question != prev_question and curr_question != []:
            prev_question = curr_question
            question_num = curr_question[0]
            
            question_text = get_question_text(task, question_num)
            answer.question_text = get_question_text(task, question_num)
                
            
            uptodate_answer = ''
            line_number = 0
            cursor_index = 0
            
            # answer = {}
            answer = Text()
            print("OBJECT", answer.col, answer.shifted, question_num)
        
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
        
        updated_text, new_line, new_cursor, shifted = update_text_and_cursor_position(uptodate_answer, shift_combined, total_lines, line_number, cursor_index, shifted, answer)
        
        
        print(f"OBJECT UPDATE Total - lines: {answer.total_lines}, row: {answer.line}, col: {answer.col}, bool: {answer.shifted}, text: {answer.text}, {shift_combined}")
        line_number = new_line
        cursor_index = new_cursor
        # print(updated_text)
        # print(f"CURRENT STRING: {accumulated_answer} OUTPUT: {answer}")
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
