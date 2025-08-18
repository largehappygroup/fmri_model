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

parser = argparse.ArgumentParser(description="Script to align timestamps of keystrokes with the fMRI file")
parser.add_argument("--computer", required=False, default='cumberland', help="This argument changes directory paths depending on whether I'm working on cumberland or my local computer")

args = parser.parse_args()

if args.computer == 'mymac':
    bass_path = "/Users/zacharykaras/Desktop"
elif args.computer == 'cumberland':
    bass_path = "/home/zachkaras/fmri"

character_path = f"{bass_path}/fmri_model/midprocessing/special_character_symbols.pkl"
with open(character_path, 'rb') as f:
    special_characters = pickle.load(f)

shift_chars_path = f"{bass_path}/fmri_model/midprocessing/shift_chars.pkl"
with open(shift_chars_path, 'rb') as f:
    shift_characters = pickle.load(f)
shift_patterns = re.compile("|".join(f"({re.escape(k)})" for k in shift_characters))
# print(shift_patterns)

##########################################################################################
############# FUNCTIONS ##################################################################
##########################################################################################

def process_keystrokes(ascii_keystrokes):
    
    # converting ascii into characters
    keystroke_chars = [chr(asci).lower() for asci in ascii_keystrokes]

    # converting special ascii characters for things like enter and shift 
    converted_chars = [special_characters[char] if char in special_characters.keys() else char for char in keystroke_chars]

    # maybe TODO: remove duplicates for shift, control, arrows

    # combining keys using shift
    converted_chars = str.join('', converted_chars)
    

    # combining shift terms and maybe TODO: remove shifts where nothing was written after
    # need to do string matching 
    def replacer(match):
        for i,key in enumerate(shift_characters, start=1):
            if match.group(i):
                return shift_characters[key]
        return match.group(0)
    
    # replace shift characters
    shift_replaced = shift_patterns.sub(replacer, converted_chars)
    # print(keystroke_chars, converted_chars, shift_replaced)
    return shift_replaced
    
    
def find_volume_keystrokes(keystroke_df, question_nums_by_volume_df, aligned_timestamp, num_vols, tr):
    
    timestep = tr*1000
    end_window = aligned_timestamp

    keystrokes_by_volume = []

    for v in range(num_vols-1, 0,-1):
        # 
        start_window = end_window - timestep

        # dense code that finds keystrokes with timestamps for current volume
        # Last steps involve converting the ascii codes into keystrokes
        idx_keystrokes_in_window = (np.where((keystroke_df['end_timestamp'] >= start_window) & (keystroke_df['end_timestamp'] < end_window)))[0]
        ascii_keystrokes = list(keystroke_df.loc[idx_keystrokes_in_window, 'ascii_code'])
        cleaned_keystrokes = process_keystrokes(ascii_keystrokes)
        # print(cleaned_keystrokes)
        # print(question_nums_by_volume_df)
        
        curr_row = question_nums_by_volume_df.loc[v]
        question_num = (np.where(curr_row == 1))[0]
        # print(question_num, curr_row)
        
        # TODO Need to find the question number by looking at end times in processed answers
        # and probably relative onset times in relative onsets, along with previous-delay info
        # Need to align the two types of timestamps...
        # can use my regressor files
        
        # question_num = list(set(keystroke_df.loc[idx_keystrokes_in_window, 'question_num']))
        # question_num = question_num[0] if len(question_num) > 0 else -1
        keystrokes_by_volume.append([v, question_num, cleaned_keystrokes])

        end_window = start_window 
    
    keystrokes_by_volume.reverse()
    clean_keystrokes_df = pd.DataFrame(keystrokes_by_volume, columns=['vol_num', 'question_num', 'keystrokes'])

    return clean_keystrokes_df

def find_question_nums_by_volume(person, task):
    regressor_base_path = f"{bass_path}/fmri_model/midprocessing/regressors/questions/{task}"
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
        # each trial was 60 seconds and I  the volumes of the fMRI scan that correspond to a given trial
        # It's consistent that the final two volumes aren't associated with a task
        final_idx = len(task_info)-1
        final_question_time = task_info.loc[final_idx, 'timestamp']
        tr_in_ms = tr*1000
        final_vol_time = final_question_time + (2*tr_in_ms)

        # Performing calculations to make the timestamp iterable by the number of volumes
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
    for person in keyfiles:
        print(person)

        # different filepaths
        onset_file = f"{keydir}/{person}/relative-onsets-{person}-{task_num}.txt"
        info_file = f"{keydir}/{person}/processed-answers-{person}-{task_num}.txt"
        
        fmri_file = f"{bass_path}/fmri_model_data/midprocess/{person}/filtered_func_data_clean.nii.gz"
        tr = 0.8 # in seconds

        keystrokes_file = f"{keydir}/{person}/keystrokes-{person}-{task_num}.txt"

        person_output_path = f"{bass_path}/fmri_model/analysis/fir/midprocess/{person}"
        os.system(f"mkdir {person_output_path}")
        
        #########################################
        ### Loading files for participant #######
        #########################################

        ### Question onsets
        try:
            onset_df = pd.read_csv(onset_file, header=None, sep=' ', names=['question_num', 'onset_time'])
            # print(onset_df.iloc[-1, 1])
        except:
            print(f"No onset file for {person}. Skipping.")
            continue

        # adding information about question number
        keystroke_df = create_keystroke_dataframe(keystrokes_file, onset_df)
        
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

        num_vols = fmri_data.header['dim'][4]

        # Aligning fMRI volumes to timestamps used for keystroke files
        aligned_timestamp = align_timestamps(task_info, num_vols, tr)
        
        # getting question numbers for each corresponding volume for annotation purposes
        question_nums_by_volume_df = find_question_nums_by_volume(person, task)
        
        # processing the raw ascii codes into something that can be interpreted by a model... probably after some more preprocessing
        cleaned_keystrokes_df = find_volume_keystrokes(keystroke_df, question_nums_by_volume_df, aligned_timestamp, num_vols, tr)

        df_outpath = f"{person_output_path}/{task}_keystrokes_by_volume.csv"
        cleaned_keystrokes_df.to_csv(df_outpath, index=False)
        # break

# The purpose of the main function is to iterate through each participants' keystroke files
# and figure out what keys were pressed during what volumes of the fMRI scan
# The output should be in a format that can be ingested by a method for creating model embeddings
# maybe a dictionary where keys are volume numbers, and keystrokes are the accumulated answer at that points
def main():
    
    keydir = f"{bass_path}/fmri_model/data"
    keyfiles = os.listdir(keydir)

    process_task('code', keydir, keyfiles)
    # process_task('prose', keydir, keyfiles)
    
            
    
if __name__ == "__main__":
    main()