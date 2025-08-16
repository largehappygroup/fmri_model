import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Script to concatenate keystrokes into discrete chunks that are more interpretable.")
parser.add_argument("--computer", required=True, default='cumberland', help="This argument changes directory paths depending on whether I'm working on cumberland or my local computer")

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
timestamps = []
# break
# ts = float(asci[0])
# timestamps.append(ts)
if asci_chr == "BACKSPACE":
    answer = answer[:-1]
elif asci_chr == "ENTER":
    answer += '\n'
else:
    answer += asci_chr


def process_participant(task, participant_path):

    df_path = f"{participant_path}/{task}_keystrokes_by_volume.csv"
    vol_keystroke_df = pd.read_csv(df_path, header=True)
    
    for i,row in vol_keystroke_df.iterrows():
        pass


def main():

    datapath = f"{bass_path}/fmri_model/analysis/fir/midprocess"
    participants = os.listdir(datapath)

    for person in participants:
        participant_path = f"{datapath}/{person}"

        process_participant('code', participant_path)
        process_participant('prose', participant_path)

if __name__ == "__main__":
    main()