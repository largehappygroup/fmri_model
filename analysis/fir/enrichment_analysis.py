import pickle
import numpy as np
import pandas as pd
from collections import Counter 
from scipy.stats import hypergeom
from collections import defaultdict
from statsmodels.stats.multitest import fdrcorrection

def run_hypergeometric(top_parcel_list, parcel_voxel_counts, n_top=10_000):
    """
    Test which Schaefer parcels are significantly enriched among the top voxels.

    Parameters
    ----------
    top_parcel_list   : list of ints — parcel label for each of the top n voxels
    parcel_voxel_counts : dict {parcel_num: total_voxel_count}
    n_top             : int — number of top voxels drawn (default 10k)

    Returns
    -------
    results : list of dicts with keys:
        parcel, observed, expected, p_raw, p_fdr, significant
    """
    N = sum(parcel_voxel_counts.values())  # total cortical voxels
    n = n_top                               # top voxels drawn

    # Count how many top voxels fell in each parcel
    observed_counts = defaultdict(int)
    for p in top_parcel_list:
        observed_counts[p] += 1

    parcels = sorted(parcel_voxel_counts.keys())

    # Compute raw p-values via hypergeometric survival function
    p_raw = []
    for parcel in parcels:
        K = parcel_voxel_counts[parcel]   # total voxels in this parcel
        k = observed_counts.get(parcel, 0)  # observed in top n
        p = hypergeom.sf(k - 1, N, K, n)   # P(X >= k)
        p_raw.append(p)

    # FDR correction across all parcels
    rejected, p_fdr = fdrcorrection(p_raw, alpha=0.05)

    results = []
    for i, parcel in enumerate(parcels):
        K = parcel_voxel_counts[parcel]
        k = observed_counts.get(parcel, 0)
        expected = (K / N) * n
        results.append({
            'parcel':      parcel,
            'observed':    k,
            'expected':    expected,
            'fold_change': k / expected if expected > 0 else np.nan,
            'p_raw':       p_raw[i],
            'p_fdr':       p_fdr[i],
            'significant': rejected[i],
        })

    return results


def filter_results(results):
    total = 0
    filtered_parcels = []
    fold_changes = []
    for r in results:
        if r['significant']:
            total += 1
            filtered_parcels.append(r['parcel'])
            fold_changes.append(r['fold_change'])
    
    return {'num_sig_parcels' : total, 'enriched_parcels' : filtered_parcels}

def translate_to_region_name(parcel_num, schaefer_labels):
        parcel_num = int(parcel_num)
        if parcel_num <= 200:
            region = schaefer_labels['left'][parcel_num]
        else:
            region = schaefer_labels['right'][parcel_num]
        return region


def main():
    best_models = ['code-deepseek_6b-ndelays_10-look_ahead_by_0',
                   'prose-starcoder2_7b-ndelays_16-look_ahead_by_3'
                   ]
    
    with open("/data/zachkaras/fmri_model_data/intermediate_results/all_results.pkl", 'rb') as f:
        records = pickle.load(f)
    
    with open("schaefer_parcel_counts.pkl", 'rb') as f:
        parcel_voxel_counts = pickle.load(f)
        
    
    schaefer_labels = "schaefer_parcel_labels.pkl"
    with open(schaefer_labels, 'rb') as f:
        region_labels = pickle.load(f)

    results = []
    for m in best_models:
        print(m)
        parts = m.split('-')
        task,model,delays,look_ahead = parts[0],parts[1],parts[2],parts[3]
        
        filtered_records = records[(records['task'] == task) & (records['model'] == model) & (records['ndelays'] == delays) & (records['look_ahead'] == look_ahead)] 
        
        for (p,layer), df in filtered_records.groupby(['participant','layer']):
            
            # Running enrichment analysis
            parcel_list = list(df['top_parcels'])[0]
            enrichment_results = run_hypergeometric(parcel_list, parcel_voxel_counts)
            significant_results = filter_results(enrichment_results)
            
            # translated enriched parcel numbers to region names from the Harvard-Oxford Atlas
            enriched_regions = [translate_to_region_name(parcel_num, region_labels) for parcel_num in significant_results['enriched_parcels']]
            enriched_regions = dict(Counter(enriched_regions))
            enriched_regions = dict(sorted(enriched_regions.items(), key=lambda x: x[1], reverse=True))

            # Concatenating results
            significant_results = {
                'model' : m,
                'task'  : task,
                'layer' : layer,
                'participant' : p,
                **significant_results,
                'enriched_region_names' : enriched_regions
            }
            
            results.append(significant_results)
        #     break
        # break
    results = pd.DataFrame(results)

    outpath = '/data/zachkaras/fmri_model_data/intermediate_results' 
    with open(f"{outpath}/enrichment_analysis_results.pkl", 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()
