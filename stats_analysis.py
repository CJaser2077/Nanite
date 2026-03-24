import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
import pingouin as pg
# supervisor : mixed effects with RT as covariate, 3 degradation x 3 speeds
# using LMM on aggregated accuracy -- avoids convergence issues with GLMM on binary data
def run_lmm_dynamic(df_agg):
    print("\nLMM: 3x3 factorial (dynamic speeds only)")
    df_dyn = df_agg[df_agg['Speed'] > 0].copy()
    df_dyn['Speed_cat'] = df_dyn['Speed'].astype(str)
    df_dyn['MPE_cat'] = df_dyn['MPE'].astype(str)
    df_dyn['RT_z'] = (df_dyn['RT_mean'] - df_dyn['RT_mean'].mean()) / df_dyn['RT_mean'].std()
    df_dyn['Acc_trans'] = np.arcsin(np.sqrt(df_dyn['Accuracy']))
    model = smf.mixedlm('Acc_trans ~ C(Speed_cat) * C(MPE_cat) + RT_z',
                        data=df_dyn, groups=df_dyn['PID'])
    result = model.fit(reml=True)
    print(result.summary())
    return result

# full model---baseline to defend against duration confounding
def run_lmm_full(df_agg):
    print("\nLMM: 4x3 full model (with static baseline)")
    df_full = df_agg.copy()
    df_full['Speed_cat'] = df_full['Speed'].astype(str)
    df_full['MPE_cat'] = df_full['MPE'].astype(str)
    df_full['RT_z'] = (df_full['RT_mean'] - df_full['RT_mean'].mean()) / df_full['RT_mean'].std()
    df_full['Acc_trans'] = np.arcsin(np.sqrt(df_full['Accuracy']))
    model = smf.mixedlm('Acc_trans ~ C(Speed_cat) * C(MPE_cat) + RT_z',
                        data=df_full, groups=df_full['PID'])
    result = model.fit(reml=True)
    print(result.summary())
    return result

# posthoc on speed and MPE separately
def run_posthoc(df_agg):
    df_dyn = df_agg[df_agg['Speed'] > 0].copy()
    df_dyn['Speed_cat'] = df_dyn['Speed'].astype(str)
    df_dyn['MPE_cat'] = df_dyn['MPE'].astype(str)
    df_dyn['Acc_trans'] = np.arcsin(np.sqrt(df_dyn['Accuracy']))
    print("\nposthoc: speed (Bonferroni)")
    df_sp = df_dyn.groupby(['PID', 'Speed_cat'])['Acc_trans'].mean().reset_index()
    ph_speed = pg.pairwise_tests(data=df_sp, dv='Acc_trans', within='Speed_cat',
                                 subject='PID', padjust='bonf')
    print(ph_speed.to_string())
    print("\nposthoc: MPE (Bonferroni)")
    df_mp = df_dyn.groupby(['PID', 'MPE_cat'])['Acc_trans'].mean().reset_index()
    ph_mpe = pg.pairwise_tests(data=df_mp, dv='Acc_trans', within='MPE_cat',
                               subject='PID', padjust='bonf')
    print(ph_mpe.to_string())
    return ph_speed, ph_mpe

# effect sizes between speed pairs
def compute_effect_sizes(df_agg):
    print("\neffect sizes (Cohen's d, speed pairs)")
    df_dyn = df_agg[df_agg['Speed'] > 0].copy()
    results = {}
    for s1, s2 in [(1.5, 5.0), (1.5, 15.0), (5.0, 15.0)]:
        g1 = df_dyn[df_dyn['Speed'] == s1].groupby('PID')['Accuracy'].mean()
        g2 = df_dyn[df_dyn['Speed'] == s2].groupby('PID')['Accuracy'].mean()
        diff = g1.values - g2.values
        d = diff.mean() / diff.std()
        results[(s1, s2)] = d
        print(f"  {s1} vs {s2}: d = {d:.3f}")
    return results

# log model fit on global thresholds // speed-degradation model
def fit_speed_degradation_model(thresholds):
    v_vals = np.array(list(thresholds.keys()))
    q_vals = np.array([d['alpha'] for d in thresholds.values()])
    log_v = np.log1p(v_vals)
    params = np.polyfit(log_v, q_vals, 1)
    q_pred = np.polyval(params, log_v)
    ss_res = np.sum((q_vals - q_pred) ** 2)
    ss_tot = np.sum((q_vals - q_vals.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"\n speed-degradation model")
    print(f"  q75(v) = {params[0]:.4f} * log(1+v) + {params[1]:.4f}")
    print(f"  R-squared = {r2:.4f}")
    return params, r2

#RT analysis supervisor: "Normality check with Shapiro-Wilk test"
def run_rt_analysis(df_agg):
    print("\n RT Normality Check (Shapiro-Wilk) ")
    df_dyn = df_agg[df_agg['Speed'] > 0].copy()
    df_rt = df_dyn.groupby(['PID', 'Speed']).agg(RT_mean=('RT_mean', 'mean')).reset_index()
    df_rt['Speed_cat'] = df_rt['Speed'].astype(str)
    norm_check = pg.normality(data=df_rt, dv='RT_mean', group='Speed_cat')
    print(norm_check.to_string())
    # supervisor: "Statistical analysis with ANOVA if data is normal"
    print("\n RT analysis (dynamic speeds: RM-ANOVA) ")
    aov = pg.rm_anova(data=df_rt, dv='RT_mean', within='Speed_cat', subject='PID', detailed=True)
    print(aov.to_string())
    ph = pg.pairwise_tests(data=df_rt, dv='RT_mean', within='Speed_cat',
                           subject='PID', padjust='bonf')
    print(ph.to_string())
    return aov, ph

# subjective vs objective correlation
def run_subjective_correlation(sub_df, mi_data):
    print("\n subjective vs objective correlation ")
    sub_df = sub_df.sort_values('PID')
    mapping = {'Vis_Low': 1.5, 'Vis_Med': 5.0, 'Vis_High': 15.0}
    for col, v in mapping.items():
        ratings = sub_df[col].values
        mi_vals = mi_data[v]
        mask = ~np.isnan(mi_vals)
        r, p = pearsonr(ratings[mask], mi_vals[mask])
        print(f"  [{col} vs MI at {v}m/s] r={r:.3f}, p={p:.4f}")