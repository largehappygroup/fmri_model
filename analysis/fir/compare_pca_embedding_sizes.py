import re
import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats



# For each participant, read in files associated with each 0 delay 
datadir = "/data2/zachkaras/fmri_model_data/fir_vectors_pca_params"

participants = os.listdir(datadir)
participants.remove('101')
participants.remove('130')

all_records = []

for p in participants:
    participant_path = f"{datadir}/{p}"
    embeddings = os.listdir(participant_path)
    filtered_embeddings = [f for f in embeddings if re.search(r".*look_ahead_by_0-ndelays_0.*regressor\+features\.pkl", f)]
    
    for f in filtered_embeddings:
        model,task,look_ahead,ndelays,_,layer,_ = f.split('-')
        emb_path = f"{participant_path}/{f}"
        with open(emb_path, 'rb') as f:
            emb = pickle.load(f)
        # print(emb.shape, f)
        record = {
            'model'      : model,
            'task'       : task,
            'layer'      : layer,
            'participant': p,
            'embedding_length': (emb.shape[1]) - 1 # subtracting one for the regressor
        }
        all_records.append(record)

record_df = pd.DataFrame.from_dict(all_records)


results = {
    'code' : {},
    'prose': {}
}   

for (task,model,participant),df in record_df.groupby(['task', 'model', 'participant']):
    length_mean = np.mean(df['embedding_length'].explode())
    results[task][participant] = length_mean

results = pd.DataFrame.from_dict(results)
t,p = stats.ttest_rel(results['code'], results['prose'])
diff = results['code'] - results['prose']
cohens_d = diff.mean() / diff.std()


print(f"the average embedding length after applying PCA for Code was {results['code'].mean():.3f}, and {results['prose'].mean():.3f} for Prose.\nWe found that this difference was statistically significantly based on a paired t-test ($t={t:.3f}$, $p={p:.3f}$, $d={cohens_d:.3f}$)")