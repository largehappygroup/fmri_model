import os
import re
import pickle
import numpy as np
import pandas as pd
import nibabel as nib

with open("/home/zachkaras/fmri/fmri_model/midprocessing/special_character_symbols.pkl", 'rb') as f:
    special_characters = pickle.load(f)


def process_answer(i, answer, timestamps, time_asci_row, num_lines):
    

    if (len(time_asci_row) <= 1 and i != 0) or (i == num_lines-1): # if it's 'new stimulus' or '<timestamp>, <ascii key>'

        # resetting the answer and timestamps after each stimulus
        answer = ''
        timestamps = []
                
    elif len(time_asci_row) == 2: # if it's the comma separated timestamp and keystroke
        
        asci_chr = chr(int(time_asci_row[1]))
        
        try: # see if it's one of the special characters
            asci_chr = special_characters[asci_chr]
        except:
            y = "carry on"
            
        ts = float(time_asci_row[0])
        timestamps.append(ts)
        if asci_chr == "BACKSPACE":
            answer = answer[:-1]
        elif asci_chr == "ENTER":
            answer += '\n'
        else:
            answer += asci_chr
    return answer, timestamps


def calculate_stim_onset(onsetfile):
    onset_df = []
    with open(onsetfile, 'r') as f:
        for line in f:
            newline = line.strip()
            onset_df.append(re.split('\s', newline))
    onset_df = pd.DataFrame(onset_df)
    onset_df.columns = ['stim_id', 'timestamp']
    # # print(onset_df)

    # print("first timestamp?", onset_df.loc[0, 'timestamp'], onset_df.loc[1, 'timestamp'])
    ts_to_match = float(onset_df.loc[1, 'timestamp'])
    
    end_ts = float(onset_df.loc[0,'timestamp'])/10**3
    alignment_time = (ts_to_match - end_ts) * (-1)
    # onset_df['timestamp'].apply(lambda x: (float(x)/10**3) - alignment_time) # this gives end timestamps
    return alignment_time


# keystrokes
def calculate_question_duration(timestamps, answer, alignment_time):
    # alignment_time = calculate_stim_onset()
    diff = (timestamps[-1] - timestamps[0])/10**3 # duration based on keystrokes
    # print('\nnew stimulus') 
    # print(diff) 
    # print([(float(t)/10**3) - alignment_time for t in timestamps])
    # print(answer) 
    return diff


def process_keystrokes(keyfile, onset_df):
    
    if not os.path.exists(keyfile):
        return pd.DataFrame()
    
    question_sequence = onset_df['question_num'].apply(lambda x: int(x))
    
    with open(keyfile, 'r') as kf:
        
        # answer = '' # where the participant's response will be accumulated
        # timestamps = [] # all the timestamps for each keystroke for a participant response
        # durations = [] # all the durations based on the keystroke data
        keystroke_df = []
        # alignment_time = calculate_stim_onset(onsetfile)
        
        qi = 0 # question number
        question = question_sequence[qi]
        
        # all_answers = {}
        # all_timestamps = {}
        
        lines = kf.readlines() # reading lines of keystroke file
        
        for i,line in enumerate(lines):
            if bool(re.search('new stimulus', line)):
                question = question_sequence[qi]
                qi += 1
                continue
            
            # as we're reading through each line of the keystroke file
            time_asci_row = re.split(', ', line.strip()) # columns are timestamp, ascii_code (e.g., 352145856.275517, 16)
            time_asci_row[0],time_asci_row[1] = float(time_asci_row[0]), int(time_asci_row[1]) # type conversion
            
            time_asci_row.insert(0, question)
            
            # print(time_asci_row)
            
            keystroke_df.append(time_asci_row)
            
            # answer, timestamps = process_answer(i, answer, timestamps, time_asci_row, num_lines=len(lines))  
                            
        keystroke_df = pd.DataFrame(keystroke_df)
        keystroke_df.columns = ['question_num', 'timestamps', 'ascii_code']

        return keystroke_df
    
def align_timestamps():
    # to align timestamps, I need to use the processed-answers files, which contain the final timestamps for each question
    pass
    
    
def make_volume_windows(num_vols, tr, keystroke_df, onset_df):
    curr_vol = 0
    time_in_ms = int(tr*1000) # converting to milliseconds since floats aren't iterable
    total_time = int(time_in_ms * num_vols)
    
    print(time_in_ms, total_time)
    print(keystroke_df)
    
    for t in range(time_in_ms, total_time, time_in_ms): # from start volume to total time, with step sizes corresponding to TR
        
        window_start = t/1000
        window_end = (t + 0.8)
        
        # iterate through keystrokes and find the ones within the range
        writing = ''
        for i,row in keystroke_df.iterrows():
            pass
            
        # for i in range(len(keystroke_df)):
            print(row)
            
            # if keystroke_df.loc[i, 'timestamps'] == 'new stimulus':
                
            #     # TODO - probably need to add some functionality here for when the stimulus changes
                
            #     continue
            
            # key_time = float(keystroke_df.loc[i, 'timestamps'])

            # # TODO - need to align times from onset file to fmri volumes
                
            # if window_end < key_time:
            #     continue
            # elif window_end > key_time and window_start < key_time:
            #     writing += keystroke_df.loc[i, 'ascii_code'] 
        
        # if writing:
        #     print(f'volume: {vol_number} | window end: {window_start} | window end: {window_end} | writing: {writing}')
            
        curr_vol += 1
        break
    
# TODO - I think there are about 16 seconds of open scan time at the start of each scan, before participants see any questions
# I should probably clip these scans to only look at brain signal for which we have corresponding data


# The purpose of the main function is to iterate through each participants' keystroke files
# and figure out what keys were pressed during what volumes of the fMRI scan
# The output should be in a format that can be ingested by a method for creating model embeddings
# maybe a dictionary where keys are volume numbers, and keystrokes are the accumulated answer at that points
def main():
    
    keydir = "/home/zachkaras/fmri/fmri_model/data"
    keyfiles = os.listdir(keydir)
    
    for person in keyfiles:
        print(person)
        onset_file = f"{keydir}/{person}/relative-onsets-{person}-3.txt"
        keystrokes_file = f"{keydir}/{person}/keystrokes-{person}-3.txt"
        
        onset_df = pd.read_csv(onset_file, header=None, sep=' ', names=['question_num', 'onset_time'])
        keystroke_df = process_keystrokes(keystrokes_file, onset_df)
        
        alignment_time = align_timestamps()
        
        if keystroke_df.empty:
            print("keytroke file doesn't exist")
            continue
        
        fmri_file = f"/home/zachkaras/fmri/fmri_model_data/midprocess/{person}/filtered_func_data_clean.nii.gz"
        fmri_data = nib.load(fmri_file)
        num_vols = fmri_data.header['dim'][4]
        tr = 0.8 # in seconds
        make_volume_windows(num_vols, tr, keystroke_df, onset_df)
        
        break
        
    
if __name__ == "__main__":
    main()

# df = []


"""
with open('../data/125/processed-answers-125-3.txt', 'r') as f:
    for line in f:
        # print(line.strip())
        newline = line.strip()
        df.append(re.split(',', newline))
df = pd.DataFrame(df)
df.columns = df.iloc[0]
df = df[1:].reset_index()

df['timestamp'].apply(lambda x: float(x)/10**3)



onset_df = []
with open('../data/125/relative-onsets-125-3.txt', 'r') as f:
    for line in f:
        newline = line.strip()
        onset_df.append(re.split('\s', newline))
onset_df = pd.DataFrame(onset_df)
onset_df.columns = ['stim_id', 'timestamp']
print(onset_df)




ts_to_match = float(onset_df.loc[1, 'timestamp'])
end_ts = float(df.loc[0,'timestamp'])/10**3
alignment_time = (ts_to_match - end_ts) * (-1)
df['timestamp'].apply(lambda x: (float(x)/10**3) - alignment_time) # this gives end timestamps


vol_number = 0
for rt in range(800, 597600, 800):
    window_start = rt/10**3
    window_end = (window_start + 0.8)
    
    # iterate through keystrokes and find the ones within the range
    writing = ''
    for i in range(len(keystroke_df)):
        key_time = float(keystroke_df.loc[i, 'timestamps'])
        if window_end < key_time:
            continue
        elif window_end > key_time and window_start < key_time:
            writing += keystroke_df.loc[i, 'ascii_code'] 
    
    if writing:
        print(f'volume: {vol_number} | window end: {window_start} | window end: {window_end} | writing: {writing}')
        
    vol_number += 1
"""
# Need to read in keystroke files



# Need to read in fMRI volumes

#  





