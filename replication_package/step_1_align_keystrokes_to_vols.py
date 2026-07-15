import os
import re
import pickle
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

##########################################################################################
############# VARIABLES ##################################################################
##########################################################################################

# parser = argparse.ArgumentParser(description="Script to align timestamps of keystrokes with the fMRI file")
# parser.add_argument("--computer", required=False, default='cumberland', help="This argument changes directory paths depending on whether I'm working on cumberland or my local computer")

# args = parser.parse_args()

# if args.computer == 'mymac':
#     bass_path = "/Users/zacharykaras/Desktop"
# elif args.computer == 'cumberland':
#     bass_path = "/home/zachkaras/fmri"
# elif args.computer == 'behemoth':
#     bass_path = "/home/zachkaras"

character_path = f"helpers/special_character_symbols.pkl"
with open(character_path, 'rb') as f:
    special_characters = pickle.load(f)
    
##########################################################################################
############# FUNCTIONS ##################################################################
##########################################################################################

def concat_duplicates(key_list):
    nonprintable_keys = [    
        '<K:S>', # shift
        '<K:BS>', # backspace
        '<K:CTRL>', # ctrl
        '<K:CTRLT>', # new tab shortcut
        '<K:ESC>', # escape
        '<K:L>', # left arrow
        '<K:R>', # right arrow
        '<K:U>', # up arrow
        '<K:D>' # down arrow
    ]
    
    concatenated_keys = []
    # iterate through the list of keystrokes
    prev_key = ''
    prev_key_count = 0
    for i,key in enumerate(key_list):
        
        # if key doesn't equal prev key - 
        # i.e. an incremented non-printable key that should be recorded or a new key
        if key != prev_key:
            
            # if there's more than one occurrence
            # format in special way
            if prev_key_count > 1:
                new_entry = f"{prev_key[:-1]} x={prev_key_count}>"
                concatenated_keys.append(new_entry)
            
            # otherwise just append key
            elif prev_key_count == 1:
                concatenated_keys.append(prev_key)
            
            # reset variables
            prev_key = ''
            prev_key_count = 0
        
        # if it's just a regular key, append it to the output
        # then reset variables
        if key not in nonprintable_keys:
            concatenated_keys.append(key.lower())
            prev_key = ''
            prev_key_count = 0
            continue
        
        # if it's a non-printable key
        else:
            # increment if it's the same as previous key
            if key == prev_key:
                prev_key_count += 1
            else: # if it's a new key, start sequence here
                prev_key = key
                prev_key_count = 1
        
        # if we reach the end of the list and have a nonempty entry, record it
        if i == len(key_list) - 1 and prev_key_count > 0:
            if prev_key_count == 1:
                concatenated_keys.append(prev_key)
            elif prev_key_count > 1:
                new_entry = f"{prev_key[:-1]} x={prev_key_count}>"
                concatenated_keys.append(new_entry)
    
    return concatenated_keys

def process_keystrokes(ascii_keystrokes):
    
    # converting ascii into characters
    keystroke_chars = [chr(asci) for asci in ascii_keystrokes]

    # converting special ascii characters for things like enter and shift 
    converted_chars = [special_characters[char] if char in special_characters.keys() else char for char in keystroke_chars]

    concatenated_chars = concat_duplicates(converted_chars)
    return concatenated_chars
    
    
def find_volume_keystrokes(keystroke_df, question_nums_by_volume_df, aligned_timestamp, num_vols, tr):
    
    timestep = tr*1000
    end_window = aligned_timestamp

    keystrokes_by_volume = []

    for v in range(num_vols-1, -1,-1):
        # 
        start_window = end_window - timestep
        print("start ", start_window, "end ", end_window)
        # dense code that finds keystrokes with timestamps for current volume
        # Last steps involve converting the ascii codes into keystrokes
        idx_keystrokes_in_window = (np.where((keystroke_df['end_timestamp'] >= start_window) & (keystroke_df['end_timestamp'] < end_window)))[0]
        ascii_keystrokes = list(keystroke_df.loc[idx_keystrokes_in_window, 'ascii_code'])
        cleaned_keystrokes = process_keystrokes(ascii_keystrokes)
        print(f"Start {start_window} | End {end_window} | Keys {cleaned_keystrokes}")
        
        curr_row = question_nums_by_volume_df.loc[v]
        question_num = (np.where(curr_row == 1))[0]
        keystrokes_by_volume.append([v, question_num, cleaned_keystrokes])

        end_window = start_window 
    
    keystrokes_by_volume.reverse()
    clean_keystrokes_df = pd.DataFrame(keystrokes_by_volume, columns=['vol_num', 'question_num', 'keystrokes'])

    return clean_keystrokes_df

def find_question_nums_by_volume(person, task):
    regressor_base_path = f"helpers/design_matrices/{task}"
    questions = [q for q in range(9)]
    
    participant_df = pd.DataFrame(columns=[i for i in range(9)])
    for q in questions:
        regressor_path = f"{regressor_base_path}/{q}/{person}.csv"
        try:
            regressor_df = pd.read_csv(regressor_path, header=None)
        except:
            continue
        participant_df[q] = regressor_df
    
    return participant_df


def align_timestamps(task_info, num_vols, tr):
    # to align timestamps, I need to use the processed-answers files, which contain the final timestamps for each question
    
    # Timestamps in processed-answers and keystrokes appear to be in milliseconds
        # subtracting the first timestamp from the last timestamp results in a value ~520,000
        # If the timestamps are in milliseconds, that means the trial would be a little less than 10 minutes
        # which makes sense since there are four blocks of questions

        # to align timestamps, the final timestamp in the processed-answers file corresponds to the 
        # time when the participant finished the final question
        # then it looks like I need to add the result of multiplying the TR by 2,
        # which corresponds to the final two volumes recorded after the end of the last stimulus.
        # I calculated 2 volumes using the following process:
        # each trial was 60 seconds and I checked that the volumes of the fMRI scan that correspond to a given trial
        # It's consistent that the final two volumes aren't associated with a task
        participant = task_info.loc[0, 'participant-id']
        tr_in_ms = tr*1000

        # participant 141 has a short functional file, but all the keystroke data
        # for that participant, the last complete answer is question 6, and there are 13 volumes collected after that
        if participant == 141:
            final_idx = np.where(task_info['stimulus-id'] == 6)[0][0]
            print(final_idx, type(final_idx))
            final_question_time = task_info.loc[final_idx, 'timestamp']
            final_vol_time = final_question_time +  (13 * tr_in_ms)
        else:
            final_idx = len(task_info)-1
            final_question_time = task_info.loc[final_idx, 'timestamp']
            final_vol_time = final_question_time + (2*tr_in_ms)

        # Performing calculations to make the timestamp iterable by the number of volumes
        # num_vols = 503 if participant == 109 else num_vols # for participant 109, based on onset times
        remainder = round(final_vol_time)%num_vols

        divisible_time = round(final_vol_time)-remainder

        return divisible_time


def create_keystroke_dataframe(keyfile, onset_df):
    
    if not os.path.exists(keyfile):
        return pd.DataFrame()
    
    question_sequence = onset_df['question_num'].apply(lambda x: int(x))
    
    with open(keyfile, 'r') as kf:
        keystroke_df = []
        
        qi = 0 # question number
        question = question_sequence[qi]
        
        lines = kf.readlines() # reading lines of keystroke file
        
        for i,line in enumerate(lines):
            if bool(re.search('new stimulus', line)):
                try:
                    question = question_sequence[qi]
                except:
                    print("Participant doesn't have data for all questions. Using questions for which we do have data.")
                    continue
                qi += 1
                continue
            
            # as we're reading through each line of the keystroke file
            time_asci_row = re.split(', ', line.strip()) # columns are timestamp, ascii_code (e.g., 352145856.275517, 16)
            time_asci_row[0],time_asci_row[1] = float(time_asci_row[0]), int(time_asci_row[1]) # type conversion
            
            time_asci_row.insert(0, question)
            keystroke_df.append(time_asci_row)
                            
        keystroke_df = pd.DataFrame(keystroke_df)
        keystroke_df.columns = ['question_num', 'end_timestamp', 'ascii_code']

        return keystroke_df
    

# iterating through participants' data based on their answers to the coding task and the prose task
# loading all the necessary data files
def process_task(task, keydir, keyfiles):

    if task == 'code':
        task_num = 3
    elif task == 'prose':
        task_num = 1

    # Iterate through each participant's data
    # 141 doesn't have the full scan, only 592 volumes
    
    # 109 had issues with the interface during prose, and the scan for code got cut short
    # For 109 code, I don't think there's any information I can use to suggest I should 
    # process the data any differently. The final timestamp in the keystrokes-3 file happens before
    # the final timestamp of the processed-answers-3 file, suggesting it was recorded. I'm guessing
    # the scan ended while the participant kept typing, so we should remove some of the final keystrokes, but I 
    # dont' have a way of saying how many volumes to offset by. 
    for person in keyfiles:

        # different filepaths
        onset_file = f"{keydir}/{person}/relative-onsets-{person}-{task_num}.txt"
        info_file = f"{keydir}/{person}/processed-answers-{person}-{task_num}.txt"
        
        if task == 'code':
            # fmri_file = f"{bass_path}/fmri_model_data/midprocess/{person}/filtered_func_data_clean.nii.gz"
            # fmri_file = f"/storage1/fmri_model_data/midprocess/{person}/filtered_func_data_clean.nii.gz"
            fmri_file = f"/data/zachkaras/fmri_model_data/midprocess/{person}/filtered_func_data_clean.nii.gz" # behemoth
        elif task == 'prose':
            # fmri_file = f"{bass_path}/fmri_model_data/midprocess_prose/{person}/filtered_func_data_clean.nii.gz"
            # fmri_file = f"/storage1/fmri_model_data/midprocess_prose/{person}/filtered_func_data_clean.nii.gz"
            fmri_file = f"/data/zachkaras/fmri_model_data/midprocess_prose/{person}/filtered_func_data_clean.nii.gz" # behemoth
        tr = 0.8 # in seconds

        keystrokes_file = f"{keydir}/{person}/keystrokes-{person}-{task_num}.txt"

        person_output_path = f"{bass_path}/fmri_model/analysis/fir/midprocess/{person}"
        if not os.path.isdir(person_output_path):
            os.system(f"mkdir {person_output_path}")
        
        #########################################
        ### Loading files for participant #######
        #########################################

        ### Question onsets
        try:
            onset_df = pd.read_csv(onset_file, header=None, sep=' ', names=['question_num', 'onset_time'])
        except:
            print(f"No onset file for {person}. Skipping.")
            continue

        # adding information about question number
        keystroke_df = create_keystroke_dataframe(keystrokes_file, onset_df)
        # keystroke_df.to_csv("test_df.csv")
        
        if keystroke_df.empty:
            print("keytroke file doesn't exist")
            continue
        
        # task info that contains final timestamps for stimuli
        try:
            task_info = pd.read_csv(info_file)
        except:
            print(f"No task file for {person}. Skipping")
            continue
        
        # fMRI data        
        try:
            fmri_data = nib.load(fmri_file)
        except:
            print(f"Cannot find fMRI data for {person}")
            continue

        actual_num_vols = int(fmri_data.header['dim'][4])
        num_vols = actual_num_vols

        if person == 109:
            num_vols = 505

        # Aligning fMRI volumes to timestamps used for keystroke files
        aligned_timestamp = align_timestamps(task_info, num_vols, tr)
        
        # getting question numbers for each corresponding volume for annotation purposes
        question_nums_by_volume_df = find_question_nums_by_volume(person, task)
        
        # processing the raw ascii codes into something that can be interpreted by a model... probably after some more preprocessing
        cleaned_keystrokes_df = find_volume_keystrokes(keystroke_df, question_nums_by_volume_df, aligned_timestamp, num_vols, tr)

        if actual_num_vols > num_vols:
            extra_rows = pd.DataFrame(
                [[v, np.array([]), []] for v in range(num_vols, actual_num_vols)],
                columns=['vol_num', 'question_num', 'keystrokes']
            )
            cleaned_keystrokes_df = pd.concat([cleaned_keystrokes_df, extra_rows], ignore_index=True)

        df_outpath = f"{person_output_path}/{task}_keystrokes_by_volume.csv"
        cleaned_keystrokes_df.to_csv(df_outpath, index=False)
        break
        

# The purpose of the main function is to iterate through each participants' keystroke files
# and figure out what keys were pressed during what volumes of the fMRI scan
# The output should be in a format that can be ingested by a method for creating model embeddings
# maybe a dictionary where keys are volume numbers, and keystrokes are the accumulated answer at that points
def main():
    
    keydir = "data"
    keyfiles = os.listdir(keydir)

    process_task('code', keydir, keyfiles)
    process_task('prose', keydir, keyfiles)           
    
if __name__ == "__main__":
    main()
