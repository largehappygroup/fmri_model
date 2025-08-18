import os
import re
import math
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


# Goal is to make discrete prompts for an LLM based on participants' keystrokes

# Need some way of concetanating keystrokes into meaningful chunks

# should start with prompt, then add on successive characters and try processing the answer

answer = ''
                
# durations.append(diff)
# timestamps = []
# break
# ts = float(asci[0])
# timestamps.append(ts)


def process_participant(task, person, participant_path):

    df_path = f"{participant_path}/{task}_keystrokes_by_volume.csv"
    try:
        vol_keystroke_df = pd.read_csv(df_path)
    except:
        print(f"No file for participant {person} for {task}")
        return
    
    # TODO need a greedy algorithm for ingesting characters
    
    answer = ''
    for i,row in vol_keystroke_df.iterrows():
        asci_chr = row['keystrokes']
        # print(type(asci_chr))
        if isinstance(asci_chr, float):
            continue
        
        # if '[BACKSPACE]' in asci_chr:
        #     print("hullo", asci_chr)
        test = re.search(r"\[BACKSPACE\]", asci_chr)
        print(type(asci_chr), test, asci_chr)
        # if re.search(asci_chr, "[BACKSPACE]"):
        #     print("hello")
        # elif asci_chr == "ENTER":
        #     answer += '\n'
        # else:
        #     answer += asci_chr


def main():

    datapath = f"{bass_path}/fmri_model/analysis/fir/midprocess"
    participants = os.listdir(datapath)

    for person in participants:
        participant_path = f"{datapath}/{person}"

        process_participant('code', person, participant_path)
        # process_participant('prose', participant_path)
        break

if __name__ == "__main__":
    main()