import os
import re
import ast
import math
import pickle
import argparse
import numpy as np
import pandas as pd

# argument that changes path names if I'm using my local computer
parser = argparse.ArgumentParser(description="Script to concatenate keystrokes into discrete chunks that are more interpretable.")
parser.add_argument("--computer", required=False, default='cumberland', help="This argument changes directory paths depending on whether I'm working on cumberland or my local computer")

args = parser.parse_args()

if args.computer == 'mymac':
    bass_path = "/Users/zacharykaras/Desktop"
elif args.computer == 'cumberland':
    bass_path = "/home/zachkaras/fmri"

# shifted alternatives of characters (s --> S, = --> +)
with open(f"{bass_path}/fmri_model/midprocessing/shift_chars.pkl", 'rb') as f:
    shift_chars = pickle.load(f)
    
code_questions = "/home/zachkaras/fmri/fmri_model/midprocessing/code_writing_prompts.csv"
code_question_df = pd.read_csv(code_questions)

prose_question = "/home/zachkaras/fmri/fmri_model/midprocessing/prose_writing_prompts.csv"
prose_question_df = pd.read_csv(prose_question)

separators = [' ', '\r', '\t', '{', '}', ',', '[', ']', '(', ')', '+', '-', ';', '=', '<=', '>=', '==', '']

###################################################################################
################### TEXT CLASS ####################################################
###################################################################################

class Text:
    """
    Editable multi-line text with a single cursor.
    - text: dictionary of strings for each line that get updated by keystrokes
    - line: current line index (0-indexed)
    - col: cursor index within line (0-indexed)
    - total_lines: total number of lines in participant's response
    - shifted: boolean indicating if the last keystroke from previous timepoint was shift (i.e., need to shift current token)
    - question_text: original question prompting participants to write code
    
    """
    def __init__(self, initial_text: str = ""):
        self.text: dict = { 0 : ''}
        self.line: int = 0 
        self.col: int = 0
        self.total_lines: int = 1
        self.shifted: bool = False
        self.question_text: str = ''

    @property
    def total_lines_count(self) -> int:
        return len(self.lines)

    @property
    def line_number(self) -> int:
        return self.line

    @property
    def cursor_index(self) -> int:
        return self.col
    
    def to_string(self) -> str:
        return '\n'.join(''.join(line) for line in self.text.values())
    
    # when a line is added (after pressing enter), need to shift all the following line numbers down
    def shift_line_numbers_down(self):
        for i in range(len(self.text.keys())-1, self.line, -1):
            self.text[i+1] = self.text[i]

    # when a line is deleted, need to shift all the following line numbers up
    def shift_line_numbers_up(self):
        for i in range(self.line, len(self.text.keys())-1):
            self.text[i] = self.text[i+1]
            
    # when participant presses enter, need to increment the line count 
    # and move all the text to the right of the cursor down to the following line
    def process_enter_key(self):
        right_string = (self.text[self.line])[self.col:]
        # all the text to the right is no longer on the current line
        self.text[self.line] = (self.text[self.line])[:self.col]
        self.shift_line_numbers_down()
        self.line += 1
        self.col = 0
        self.total_lines += 1
        self.text[self.line] = right_string
    
    # when left arrow keys are pressed, moving the cursor to the left, and potentially to the line above if it exists
    def process_left_arrows(self, left_shift):
        new_col_num = self.col - left_shift
        if new_col_num < 0:
            if self.line == 0:
                self.col = 0
            else:
                left_shift_remainder = left_shift - self.col
                self.line -= 1
                self.col = len(self.text[self.line]) - left_shift_remainder
        else:
            self.col = new_col_num
    
    # when right arrow keys are pressed, moving cursor to the right, and potentially to the line below if it exists
    def process_right_arrows(self, right_shift):
        new_col_num = self.col + right_shift
        curr_line_length = len(self.text[self.line])
        
        # if cursor is at the end of the current line
        if new_col_num > curr_line_length:
            if self.line == (self.total_lines) - 1:
                self.col = curr_line_length
            # if there are no other lines below, do nothing
            else:
                right_shift_remainder = curr_line_length - self.col
                self.line += 1
                self.col = right_shift_remainder
            # go to next line, and change col to 0
        else:
            self.col = new_col_num
    
    # moving cursor up if there are lines above
    def process_up_arrows(self, up_shift):
        new_line_num = self.line - up_shift 
        
        if new_line_num < 0:
            self.line = 0
        else:
            self.line = new_line_num
            
    # moving cursor down if there are lines below
    def process_down_arrows(self, down_shift):
        new_line_num = self.line + down_shift
        
        if new_line_num > (self.total_lines) - 1:
            self.line = (self.total_lines) - 1
        else:
            self.line = new_line_num
    
    # processing delete key
    def process_one_backspace(self):
        
        # deleting character to the left of the cursor
        left_string = (self.text[self.line])[:self.col-1]
        
        # keeping track of characters to the right of cursor
        right_string = (self.text[self.line])[self.col:]
        
        # concatenating new string
        self.text[self.line] = left_string + right_string
        
        curr_line_length = len(self.text[self.line])
        if self.col == 0:
            if curr_line_length == 0:
                prev_line_length = 0 if self.line == 0 else len(self.text[self.line-1])
                if self.line == 0:
                    combined_text = self.text[self.line]
                    self.text[0] = combined_text
                else:
                    combined_text = self.text[self.line-1] + self.text[self.line]
                    self.text[self.line-1] = combined_text
                self.col = prev_line_length - 1 if prev_line_length > 0 else 0
                self.total_lines = max(1, self.total_lines - 1)
                self.line -= 1 if self.line > 0 else 0

        else:
            self.col -= 1
    
    # decrementing text, line numbers, and cursor position for each occurrence of backspace key
    def process_multiple_backspaces(self, del_count):
        while del_count > 0:
            if self.col == 0:
                if self.line == 0:
                    break
                else:
                    if len(self.text[self.line]) == 0:
                        self.shift_line_numbers_up()
                    self.line -= 1
                    self.col = len(self.text[self.line])-1
            new_left_string = (self.text[self.line])[:self.col-1]
            right_string = (self.text[self.line])[self.col:]
            self.text[self.line] = new_left_string + right_string
            del_count -= 1
            self.col -= 1
    
    # replacing a token with shifted version
    def shift_token(self, shifted_token):
        self.text[self.line] += shifted_token
        self.col += len(shifted_token)
        self.shifted = False
    
    # if it's just a normal token, append it to current cursor position
    def append_token(self, token):
        left_string = (self.text[self.line])[:self.col] + token
        right_string = (self.text[self.line])[self.col:]
        new_string = left_string + right_string
        self.text[self.line] = new_string
        self.col += len(token)
                

##############################################################################
####################### FUNCTIONS ############################################
##############################################################################

def update_text_and_cursor_position(new_text, answer):

    for i,token in enumerate(new_text):
        one_bs_match   = re.search('<K:BS>', token)
        many_bs_match  = re.search(r'<K:BS x=([0-9]+)>', token)
        ctrl_match  = re.search('<K:CTRL', token)
        shift_match = re.search('<K:S', token)
        
        left_match  = re.search(r'(<K:L>)|(<K:L x=([0-9]+))', token)
        right_match = re.search(r'(<K:R>)|(<K:R x=([0-9]+))', token)
        up_match    = re.search(r'(<K:U>)|(<K:U x=([0-9]+))', token)
        down_match  = re.search(r'(<K:D>)|(<K:D x=([0-9]+))', token)
        
        if re.search(r'\t', token):
            answer.col += 4
            answer.text[answer.line] += '    '
            continue
        
        if re.search(r'\r', token):
            answer.process_enter_key()
            continue
            
        if left_match:
            left_shift  = 1 if not left_match.group(3) else int(left_match.group(3))
            answer.process_left_arrows(left_shift)
            continue
            
        elif right_match:
            right_shift = 1 if not right_match.group(3) else int(right_match.group(3))
            answer.process_right_arrows(right_shift)
            continue
            
        elif up_match:
            up_shift = 1 if not up_match.group(3) else int(up_match.group(3))
            answer.process_up_arrows(up_shift)
            continue
            
        elif down_match:
            down_shift  = 1 if not down_match.group(3) else int(down_match.group(3))
            answer.process_down_arrows(down_shift)
            continue
        
        if ctrl_match or shift_match:
            if i == len(new_text)-1 and shift_match: # if the last token is a shift key
                answer.shifted = True
            continue

        if one_bs_match:
            answer.process_one_backspace()
            
        elif many_bs_match:
            del_count = int(many_bs_match.group(1))
            answer.process_multiple_backspaces(del_count)

        elif answer.shifted:
            try:
                shifted_token = shift_chars[token]
            except:
                answer.text[answer.line] += token    
                answer.col += len(token) 
                continue
            answer.shift_token(shifted_token)
        else:
            answer.append_token(token)

def fill_in_the_middle(text, prefix, suffix):
    
    # prefix should be the question text and the keystrokes so far
    # wrap in <PRE><SUF><MID>
    # each model has their own conventions, but I'll replace those at the time of prompting
    # suffix should be the remainder of current token ('ate' for isDuplic(ate) )
    formatted_text = f"<PRE>{prefix}<SUF>{suffix}<MID>{text}"
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
                    # print(f"No entry for {next_key}, {ascii(next_key)}")
                    continue
                combined_text.append(shifted_char)
                shifted = True
            else:
                combined_text.append(t)       
        else:
            combined_text.append(t)
            
    return combined_text

def get_question_text(task, question_num):
    question_df = code_question_df if task == 'code' else prose_question_df
    
    question_idx = np.where(question_df['stim_id'] == question_num)
    question_text = (list(question_df.loc[question_idx, 'text']))[0]
    
    return question_text

def process_keystrokes(vol_keystroke_df, task, person):
    
    output = {i : '' for i in range(len(vol_keystroke_df))}
    only_new_keystrokes = {i : '' for i in range(len(vol_keystroke_df))}
    
    prev_question = -1
    answer = Text()
    
    for i,row in vol_keystroke_df.iterrows():
        curr_question = ast.literal_eval(row['question_num'])

        # line number and cursor_index should reset with each question
        if curr_question != prev_question and curr_question != []:
            prev_question = curr_question
            question_num = curr_question[0]
            
            # creating new text object to contain participant's response to current question
            answer = Text()
            answer.question_text = get_question_text(task, question_num)
        
        vol_text = ast.literal_eval(row['keystrokes'])
        
        if len(vol_text) == 0:
            if row['question_num'] != '[]':
                output[i] = f"{answer.question_text}\n{answer.to_string()}"
            continue
        
        # combining shift sequences
        shift_combined = combine_shift_sequences(vol_text)
        
        if shift_combined[-1] in separators:    
            next_text = ''
        else:
            next_text = find_next_sequence(i, vol_keystroke_df)
        
        prefix = f"{answer.question_text}\n{answer.to_string()}"
        formatted_text = fill_in_the_middle(''.join(shift_combined), prefix, next_text)
        output[i] = formatted_text
        only_new_keystrokes[i] = f"{''.join(shift_combined)}{next_text}" 
        
        # print(formatted_text)
        
        update_text_and_cursor_position(shift_combined, answer)
    
        # print(f"UPDATE Total - lines: {answer.total_lines}, row: {answer.line}, col: {answer.col}, bool: {answer.shifted}, text: {answer.text}, {shift_combined}")
    output_path = f"{bass_path}/fmri_model/analysis/fir/midprocess/{person}/{task}_formatted_keystrokes.pkl"
    new_keystroke_outpath = f"{bass_path}/fmri_model/analysis/fir/midprocess/{person}/{task}_new_keystrokes.pkl"
    # with open(output_path, 'wb') as f:
    #     pickle.dump(output, f)
    with open(new_keystroke_outpath, 'wb') as f:
        pickle.dump(only_new_keystrokes, f)

def process_participant(task, person, participant_path):

    df_path = f"{participant_path}/{task}_keystrokes_by_volume.csv"
    try:
        vol_keystroke_df = pd.read_csv(df_path)
    except:
        print(f"No file for participant {person} for {task}")
        return
    
    process_keystrokes(vol_keystroke_df, task, person)

def main():

    datapath = f"{bass_path}/fmri_model/analysis/fir/midprocess"
    participants = os.listdir(datapath)

    for person in participants:
        print(person)
        participant_path = f"{datapath}/{person}"

        process_participant('code', person, participant_path)
        process_participant('prose', person, participant_path)
        # break

if __name__ == "__main__":
    main()
