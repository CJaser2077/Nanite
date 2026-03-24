import data_loader
import psych_fitting
import stats_analysis
import plotter
import performance_projection
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main():
    print("data analysis start!!!\n")
    #load
    df = data_loader.load_and_preprocess('UE5_behavioral.csv')
    sub_df = data_loader.load_subjective_data('UE5_subjective.csv')
    nanite_stats = data_loader.load_nanite_stats('nanite_stats.csv')
    print(f"speeds: {sorted(df['Speed'].unique())}")
    print(f"MPE levels: {sorted(df['MPE'].unique())}")

    #Table 5.1
    print("\n[Table 5.1] clip duration sanity check")
    for spd in sorted(df['Speed'].unique()):
        dur = df[df['Speed'] == spd]['Effective_Duration_s']
        print(f"  speed {spd:5.1f}: {dur.mean():.4f}s (SD={dur.std():.4f})")

    #Table 5.2
    print("\n[Table 5.2] mean accuracy per condition")
    acc_tbl = df.groupby(['Speed', 'MPE'])['Correct'].mean().unstack()
    print(acc_tbl.round(4).to_string())

    #LMM
    df_agg = psych_fitting.aggregate_by_condition(df)
    print(f"\naggregated: {df_agg.shape[0]} cells ({df_agg['N_trials'].iloc[0]} trials each)")
    print("\n[Table 5.3] global psychometric fit")
    global_thresholds = psych_fitting.get_global_thresholds(df)
    for v in sorted(global_thresholds):
        p = global_thresholds[v]
        print(f"  speed {v}: q75 = {p['alpha']:.3f}, slope = {p['beta']:.3f}")

    #motion masking index
    print("\n[MI] accuracy-based masking index")
    mi_data = psych_fitting.compute_mi_from_accuracy(df_agg)
    for v in [1.5, 5.0, 15.0]:
        vals = mi_data[v]
        clean = vals[~np.isnan(vals)]
        print(f"  speed {v}: MI = {np.mean(clean)*100:.1f}% (SD={np.std(clean)*100:.1f}%)")

    #supervisor want mixed effects, Speed*MPE interaction, RT as covariate
    print("\n[LMM] mixed-effects analysis")
    # 1. 3x3 Model For supervisor request
    print("  (3x3 dynamic conditions)")
    lmm_dyn = stats_analysis.run_lmm_dynamic(df_agg)
    # 2. 4x3 Model For Baseline comparison
    print("\n  (4x3 full model incl. baseline)")
    lmm_full = stats_analysis.run_lmm_full(df_agg)
    # RT Shapiro-Wilk
    print("\n[RT] response time analysis")
    rt_aov, rt_ph = stats_analysis.run_rt_analysis(df_agg)
    #posthoc comparisons
    print("\n[posthoc]")
    ph_speed, ph_mpe = stats_analysis.run_posthoc(df_agg)
    #effect sizes
    print("\n[effect sizes]")
    stats_analysis.compute_effect_sizes(df_agg)
    # speed-degradation log model
    print("\n[model fit]")
    model_params, r2 = stats_analysis.fit_speed_degradation_model(global_thresholds)
    #RT analysis
    print("\n[RT analysis]")
    stats_analysis.run_rt_analysis(df_agg)

    #Table 5.5
    print("\n[Table 5.5] questionnaire summary")
    print(f"  SSQ:      {sub_df['SSQ'].mean():.2f} (SD={sub_df['SSQ'].std():.2f})")
    print(f"  NASA-TLX: {sub_df['NASA_TLX'].mean():.2f} (SD={sub_df['NASA_TLX'].std():.2f})")
    for col in ['Vis_Static', 'Vis_Low', 'Vis_Med', 'Vis_High']:
        print(f"  {col}: {sub_df[col].mean():.2f} (SD={sub_df[col].std():.2f})")
    #subjective vs objective
    stats_analysis.run_subjective_correlation(sub_df, mi_data)
    # nanite projection table (boss: report nanite triangle findings)
    if nanite_stats is not None:
        performance_projection.generate_projection_table('nanite_stats.csv',
                                                          model_params[0], model_params[1])
    #plots
    plotter.plot_all(df, df_agg, global_thresholds, mi_data, model_params, nanite_stats)
    #summary
    print("\n data analysis completed!!!")

if __name__ == '__main__':
    main()
