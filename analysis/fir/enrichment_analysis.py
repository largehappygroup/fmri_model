import os
import pickle
import numpy as np
from scipy.stats import hypergeom
from collections import defaultdict
from statsmodels.stats.multitest import fdrcorrection

# def nested_dict():
#     return defaultdict(nested_dict)

    
# with open("results/no_regressor-top_parcels_per_participant.pkl", 'rb') as f:
#     parcel_voxel_counts = pickle.load(f)

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
        # print(f"K: {K} | k: {k} | N: {N} | n: {n}")
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



def main():
    best_models = ['code-deepseek_6b-ndelays_4-look_ahead_by_10', 
            #    'code-codegemma_7b-ndelays_4-look_ahead_by_10',
            #    'code-codegemma_7b-ndelays_16-look_ahead_by_0', 
                'prose-deepseek_6b-ndelays_4-look_ahead_by_10'
            #    'prose-codegemma_7b-ndelays_10-look_ahead_by_5', 
            #    'prose-deepseek_6b-ndelays_20-look_ahead_by_10'
            ]
    with open("/data/zachkaras/fmri_model_data/intermediate_results/all_results.pkl", 'rb') as f:
        records = pickle.load(f)

    for m in best_models:
        print(m)
        parts = m.split('-')
        task,model,delays,look_ahead = parts[0],parts[1],parts[2],parts[3]
        
        filtered_records = records[(records['task'] == task) & (records['model'] == model) & (records['ndelays'] == delays) & (records['look_ahead'] == look_ahead)]
        
        # participants = set(filtered_records['participant']) 
        
        for (p,layer), df in filtered_records.groupby(['participant','layer']):
            print(p, layer)
            print(df.head)
            
            
            # break
        break
        
        
    #     for p in participants:
    #         # print(p)
    #         top_parcels = parcel_collection[p][task][model][delays][look_ahead]
            
    #         if not top_parcels:
    #             print(f"Skipping {vip} {m}: no data")
    #             continue
            
    #         # can iterate through different layers here
    #         for layer,parcel_list in top_parcels.items():
    #             # print(layer, p)
    #             enrichment_results = run_hypergeometric(parcel_list, parcel_voxel_counts)
    #             significant_results = filter_results(enrichment_results)
    #             # print(enrichment_results)
    #             participant_result_dictionary[m][layer][p] = significant_results
    #             # participant_result_dictionary[m][layer][p] = enrichment_results
    #     #         break
    #     #     break
    #     # break
    #         # plot_top_parcels(top_parcels, vip, m)


    # for m,l_p in participant_result_dictionary.items():
    #     for l,p_results in l_p.items():
    #         for p, results in p_results.items():
    #             print(p, m, l, results)

    # with open('results/participant_parcel_enrichment_results.pkl', 'wb') as f:
    #     pickle.dump(participant_result_dictionary, f)


    # with open('results/participant_parcel_enrichment_results.pkl', 'rb') as f:
    #     result_dict = pickle.load(f)


    # schaefer_labels = "schaefer_parcel_labels.pkl"
    # with open(schaefer_labels, 'rb') as f:
    #     region_labels = pickle.load(f)

    # def translate_to_region_name(parcel_num):
    #     parcel_num = int(parcel_num)
    #     if parcel_num < 200:
    #         region = schaefer_labels['left'][parcel_num]
    #     else:
    #         region = schaefer_labels['right'][parcel_num]
    #     return region

    # from collections import Counter                                                                                                                                                                                                                                                                                  
    
    # # counts = Counter(my_list)  

    # for m,l_p in result_dict.items():
    #     for l,p_results in l_p.items():
    #         for p, results in p_results.items():
    #             enriched = results['enriched_parcels']
    #             enriched_regions = [translate_to_region_name(parcel_num) for parcel_num in enriched]
    #             print(p, enriched_regions)
    #             # print(p, m, l, results)
    #             # for parcel in enriched parcels
    #             # translate 

if __name__ == "__main__":
    main()