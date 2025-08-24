import re
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

# Optional: provide a default SHIFT map if one is not passed in
DEFAULT_SHIFT_CHARS = {
    '1': '!', '2': '@', '3': '#', '4': '$', '5': '%', '6': '^',
    '7': '&', '8': '*', '9': '(', '0': ')',
    '-': '_', '=': '+', '[': '{', ']': '}', '\\': '|',
    ';': ':', "'": '"', ',': '<', '.': '>', '/': '?', '`': '~'
    # letters handled by .upper()
}

separators = [' ', '\r', '\t', '{', '}', ',', '[', ']', '(', ')',
              '+', '-', ';', '=', '<=', '>=', '==', '']

def _apply_shift_char(ch: str, shift_map: Optional[Dict[str, str]] = None) -> str:
    if not shift_map:
        shift_map = DEFAULT_SHIFT_CHARS
    if ch in shift_map:
        return shift_map[ch]
    if len(ch) == 1 and ch.isalpha():
        return ch.upper()
    return ch

class Text:
    """
    Editable multi-line text with a single cursor.
    - lines: internal list of list[str] (per-line characters)
    - line: current line index (1-based externally, 0-based internally)
    - col: cursor index within line (0-based)
    """
    def __init__(self, initial_text: str = ""):
        raw_lines = initial_text.split('\r') if initial_text else [""]
        self.lines: List[List[str]] = [list(l) for l in raw_lines]
        self.line: int = 1  # 1-based
        self.col: int = len(self.lines[0]) if self.lines else 0
        self.goal_col: Optional[int] = None  # For Up/Down column memory

    # ------------ basic getters -------------
    @property
    def total_lines(self) -> int:
        return len(self.lines)

    @property
    def line_number(self) -> int:
        return self.line  # 1-based

    @property
    def cursor_index(self) -> int:
        return self.col  # 0-based

    def _cur_line_chars(self) -> List[str]:
        return self.lines[self.line - 1]

    def to_string(self) -> str:
        return '\r'.join(''.join(line) for line in self.lines)

    def as_left_right_dict(self) -> Dict[int, List[List[str]]]:
        """
        Returns: { line_number: [left_chars, right_chars] }
        For non-current lines, the right list is empty (cursor isn't there).
        """
        view = {}
        for i, line in enumerate(self.lines, start=1):
            if i == self.line:
                left = line[:self.col]
                right = line[self.col:]
                view[i] = [left, right]
            else:
                view[i] = [line[:], []]
        return view

    # ------------ editing ops -------------
    def insert_char(self, ch: str):
        if ch == '\r':
            self.newline()
        else:
            cur = self._cur_line_chars()
            cur.insert(self.col, ch)
            self.col += 1
            self.goal_col = self.col

    def insert_text(self, s: str):
        for ch in s:
            self.insert_char(ch)

    def newline(self):
        cur = self._cur_line_chars()
        left = cur[:self.col]
        right = cur[self.col:]
        # split line
        self.lines[self.line - 1] = left
        self.lines.insert(self.line, right)
        self.line += 1
        self.col = 0
        self.goal_col = 0

    def backspace(self, n: int = 1):
        while n > 0:
            if self.col > 0:
                cur = self._cur_line_chars()
                del cur[self.col - 1]
                self.col -= 1
            else:
                # merge with previous line if exists
                if self.line == 1:
                    break
                prev_line_chars = self.lines[self.line - 2]
                cur = self._cur_line_chars()
                prev_len = len(prev_line_chars)
                prev_line_chars.extend(cur)
                del self.lines[self.line - 1]
                self.line -= 1
                self.col = prev_len
            n -= 1
        self.goal_col = self.col

    def move_left(self, n: int = 1):
        while n > 0:
            if self.col > 0:
                self.col -= 1
            else:
                if self.line == 1:
                    break
                self.line -= 1
                self.col = len(self._cur_line_chars())
            n -= 1
        self.goal_col = self.col

    def move_right(self, n: int = 1):
        while n > 0:
            cur_len = len(self._cur_line_chars())
            if self.col < cur_len:
                self.col += 1
            else:
                if self.line >= self.total_lines:
                    break
                self.line += 1
                self.col = 0
            n -= 1
        self.goal_col = self.col

    def _move_vertical(self, delta: int, n: int = 1):
        # delta: -1 up, +1 down
        goal = self.goal_col if self.goal_col is not None else self.col
        while n > 0:
            new_line = self.line + delta
            if not (1 <= new_line <= self.total_lines):
                break
            self.line = new_line
            self.col = min(goal, len(self._cur_line_chars()))
            n -= 1
        self.goal_col = goal

    def move_up(self, n: int = 1):
        self._move_vertical(-1, n)

    def move_down(self, n: int = 1):
        self._move_vertical(+1, n)
        
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


def update_text_and_cursor_position(
    text_obj: Text,
    new_tokens: List[str],
    shifted: bool = False,
    shift_chars: Optional[Dict[str, str]] = None
) -> Tuple[str, int, int, bool]:
    """
    Applies new_tokens to text_obj, updates cursor by arrow keys, handles backspace and returns/enter.
    Returns (updated_text_string, line_number(1-based), cursor_index(0-based), shifted_flag).
    """
    for i, token in enumerate(new_tokens):
        if not token:
            continue

        # Special keys
        m = re.match(r'<K:BS(?: x=(\d+))?>', token)
        if m:
            n = int(m.group(1)) if m.group(1) else 1
            text_obj.backspace(n)
            continue

        m = re.match(r'<K:L(?: x=(\d+))?>', token)
        if m:
            n = int(m.group(1)) if m.group(1) else 1
            text_obj.move_left(n)
            continue

        m = re.match(r'<K:R(?: x=(\d+))?>', token)
        if m:
            n = int(m.group(1)) if m.group(1) else 1
            text_obj.move_right(n)
            continue

        m = re.match(r'<K:U(?: x=(\d+))?>', token)
        if m:
            n = int(m.group(1)) if m.group(1) else 1
            text_obj.move_up(n)
            continue

        m = re.match(r'<K:D(?: x=(\d+))?>', token)
        if m:
            n = int(m.group(1)) if m.group(1) else 1
            text_obj.move_down(n)
            continue

        # Modifiers (Shift/Control)
        if re.match(r'<K:S', token):
            # Carry shift to the next token if this is the last token in the batch
            if i == len(new_tokens) - 1:
                shifted = True
            continue

        if re.match(r'<K:CTRL', token):
            # Not altering text for CTRL in this implementation
            continue

        # Regular text (possibly contains carriage returns)
        if '\r' in token:
            for ch in token:
                if ch == '\r':
                    text_obj.newline()
                else:
                    if shifted:
                        text_obj.insert_char(_apply_shift_char(ch, shift_chars))
                        shifted = False
                    else:
                        text_obj.insert_char(ch)
        else:
            if shifted:
                for ch in token:
                    text_obj.insert_char(_apply_shift_char(ch, shift_chars))
                shifted = False
            else:
                for ch in token:
                    text_obj.insert_char(ch)

    return text_obj.to_string(), text_obj.line_number, text_obj.cursor_index, shifted


def process_keystrokes(vol_keystroke_df, task):
    prev_question = -1
    text_obj = None
    shifted = False

    for i, row in vol_keystroke_df.iterrows():
        curr_question = ast.literal_eval(row['question_num'])

        if curr_question != prev_question and curr_question != []:
            prev_question = curr_question
            question_num = curr_question[0]
            question_text = get_question_text(task, question_num)

            text_obj = Text("")  # start fresh for this question
            shifted = False

        vol_text = ast.literal_eval(row['keystrokes'])
        if not vol_text:
            continue

        shift_combined = combine_shift_sequences(vol_text)

        if shift_combined[-1] in separators:
            next_text = ''
        else:
            next_text = find_next_sequence(i, vol_keystroke_df)

        prefix = f"{question_text} {text_obj.to_string()}"
        formatted_text = fill_in_the_middle(str.join('', shift_combined), prefix, next_text)

        updated_text, line_no, cursor_idx, shifted = update_text_and_cursor_position(
            text_obj, shift_combined, shifted, shift_chars  # shift_chars is your mapping
        )

        # If you need the dictionary view {line: [left, right]}
        per_line_view = text_obj.as_left_right_dict()

        print(updated_text)
        
        
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
        break

if __name__ == "__main__":
    main()