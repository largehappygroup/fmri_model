import os
import re
import ast
import pickle
import numpy as np
import pandas as pd


participants = os.listdir("midprocess")

def identify_outliers(task, p, keys, characters):
    mean = np.mean(keys)
    extreme_val = np.std(keys) * 5
    outlier_loc = list(np.where(keys > extreme_val + mean)[0])

    filtered_outliers = []
    for idx in outlier_loc:
        keys_pressed = ast.literal_eval(characters.loc[idx, 'keystrokes'])
        if len(keys_pressed) <= 2: # finding keys pressed that aren't just repeated backspaces or arrow keys
            continue
        elif extreme_val - len(keys_pressed) > 0: # another check for keystrokes that are mostly non-printable keys
            continue
        else: # fishy keystrokes, but care about keystrokes that are from hardware issues where keys just get repeated
            unique_keys = set(keys_pressed)
            if len(unique_keys) <= 2: # finding sequences that are repeated keystrokes
                filtered_outliers.append(idx)
    return filtered_outliers


outlier_vols = {
    'code' : {},
    'prose': {}
}
for task in ['code', 'prose']:
    for p in participants:
        try:
            with open(f"midprocess/{p}/{task}_num_keystrokes_regressor.pkl", 'rb') as f:
                keys = pickle.load(f)
        except:
            continue

        # cross reference with keystrokes file
        characters_file = f"midprocess/{p}/{task}_keystrokes_by_volume.csv"
        characters = pd.read_csv(characters_file)

        filtered_outliers = identify_outliers(task, p, keys, characters)
        if filtered_outliers:
            outlier_vols[task][p] = filtered_outliers


outpath = "/home/zachkaras/fmri_model/analysis/fir/outlier_volumes.pkl"

with open(outpath, 'wb') as f:
    pickle.dump(outlier_vols, f)
