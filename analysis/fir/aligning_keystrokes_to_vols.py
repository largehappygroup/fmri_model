import os
import re
import pickle
import numpy as np
import pandas as pd
import nibabel as nib

with open("/home/zachkaras/fmri/fmri_model/midprocessing/special_character_symbols.pkl", 'rb') as f:
    special_characters = pickle.load(f)


def process_answer(answer):
    #SHIFT9, SHIFT0, SHIFT5, SHIFT[, SHIFT]
    pass


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


def process_keystrokes(keyfile, onsetfile):
    
    if not os.path.exists(keyfile):
        return pd.DataFrame()
    
    with open(keyfile, 'r') as kf:
        
        answer = '' # where the participant's response will be accumulated
        timestamps = [] # all the timestamps for each keystroke for a participant response
        durations = [] # all the durations based on the keystroke data
        keystroke_df = []
        alignment_time = calculate_stim_onset(onsetfile)
        # return
        
        lines = kf.readlines() # reading lines of keystroke file
        
        for i, line in enumerate(lines):
            
            
            # as we're reading through each line of the keystroke file
            time_asci_row = re.split(',', line.strip()) # columns are timestamp, ascii_code (e.g., 352145856.275517, 16)
            keystroke_df.append(time_asci_row) # adding these two columns to keystroke_df 
            # what is the purpose of keystroke_df if I'm just adding a row to the text file?
            
            
            if (len(time_asci_row) <= 1 and i != 0) or (i == len(lines) - 1): # if it's 'new stimulus' or '<timestamp>, <ascii key>'
            # if (len(time_asci_row) <= 1) or (i == len(lines) - 1): # if it's 'new stimulus' or '<timestamp>, <ascii key>'
                question_dur = calculate_question_duration(timestamps, answer, alignment_time)              #      or the first or last stimulus
                
                # print(diff)
                
                test = process_answer(answer)
                answer = ''
                
                durations.append(question_dur)
                timestamps = []
                # break
            elif len(time_asci_row) == 2: # if it's the comma separated timestamp and keystroke
                asci_chr = chr(int(time_asci_row[1]))
                # asci_chr = chr(asci_int)
                
                try:
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
                            
        keystroke_df = pd.DataFrame(keystroke_df)
        keystroke_df.columns = ['timestamps', 'ascii_code']
        # keystroke_df = keystroke_df.loc[1:]
        return keystroke_df

    
# TODO - I think there are about 16 seconds of open scan time at the start of each scan, before participants see any questions
# I should probably clip these scans to only look at brain signal for which we have corresponding data

def main():
    keydir = "/home/zachkaras/fmri/fmri_model/data"
    keyfiles = os.listdir(keydir)
    for person in keyfiles:
        keystrokes_file = f"{keydir}/{person}/keystrokes-{person}-3.txt"
        onset_file = f"{keydir}/{person}/relative-onsets-{person}-3.txt"
        
        keystroke_df = process_keystrokes(keystrokes_file, onset_file)
        
        if keystroke_df.empty:
            print("keytroke file doesn't exist")
            continue
        
        vol_number = 0
        for rt in range(800, 597600, 800): # TODO - figure out what this hard coded value is and why it's this
            window_start = rt/10**3
            window_end = (window_start + 0.8)
            
            # iterate through keystrokes and find the ones within the range
            writing = ''
            for i in range(len(keystroke_df)):
                
                if keystroke_df.loc[i, 'timestamps'] == 'new stimulus':
                    
                    # TODO - probably need to add some functionality here for when the stimulus changes
                    
                    continue
                
                key_time = float(keystroke_df.loc[i, 'timestamps'])

                # TODO - need to align times from onset file to fmri volumes
                    
                if window_end < key_time:
                    continue
                elif window_end > key_time and window_start < key_time:
                    writing += keystroke_df.loc[i, 'ascii_code'] 
            
            if writing:
                print(f'volume: {vol_number} | window end: {window_start} | window end: {window_end} | writing: {writing}')
                
            vol_number += 1
        break
        
        
        # reading in keystroke file from long response coding
        # try: # fill in the blank is category 2, long response is category 3
        # except:
        #     print("no keystroke files")
        #     continue
        
    
if __name__ == "__main__":
    main()

df = []


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





