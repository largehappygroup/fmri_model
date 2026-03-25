import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

predictor_cols = [
    'GPA',
    'age',
    'years_experience',
    'code_editing_rate',
    'prose_editing_rate',
    'code_total_keystrokes',
    'prose_total_keystrokes',
    'code_more_structured',  # binary, not standardized
    'num_enriched_parcels',
]

for m in best_models:
    layers = sorted(participant_means[m].keys())

    # average performance across layers
    all_pids = set(pid for layer_num in layers for pid in participant_means[m][layer_num])
    layer_averaged_perf = {
        pid: float(np.mean([participant_means[m][ln][pid] for ln in layers if pid in participant_means[m][ln]]))
        for pid in all_pids
    }

    # average enriched parcel counts across layers
    enrich_by_pid = {}
    for layer_num in layers:
        layer_key = f'layer_{layer_num}'
        if layer_key not in enrichment_results[m]:
            continue
        for pid_str, val in enrichment_results[m][layer_key].items():
            enrich_by_pid.setdefault(int(pid_str), []).append(val['num_sig_parcels'])
    layer_averaged_enrich = {pid: float(np.mean(vals)) for pid, vals in enrich_by_pid.items()}

    # build dataframe aligned on common participants
    common_ids = [pid for pid in layer_averaged_perf if pid in demo_data.index]
    df_model = demo_data.loc[common_ids, [c for c in predictor_cols if c != 'num_enriched_parcels']].copy()
    df_model['num_enriched_parcels'] = [layer_averaged_enrich.get(pid, float('nan')) for pid in common_ids]
    df_model['brain_perf'] = [layer_averaged_perf[pid] for pid in common_ids]
    df_model = df_model.dropna()

    # standardize continuous predictors (not the binary one)
    continuous = [c for c in predictor_cols if c != 'code_more_structured']
    scaler = StandardScaler()
    df_model[continuous] = scaler.fit_transform(df_model[continuous])

    X = sm.add_constant(df_model[predictor_cols])
    y = df_model['brain_perf']

    result = sm.OLS(y, X).fit()

    print("=" * 70)
    print(f"Model: {m}  |  N = {len(df_model)}")
    print("=" * 70)
    print(result.summary())
    print()