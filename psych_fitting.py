import numpy as np
from scipy.optimize import curve_fit

# standard 2AFC logistic
def logistic_func(x, alpha, beta, gamma=0.5, lamb=0.01):
    return gamma + (1 - gamma - lamb) * (1 / (1 + np.exp(-(x - alpha) / beta)))
def get_global_thresholds(df):
    # fit one curve per speed
    speeds = sorted(df['Speed'].unique())
    results = {}
    for v in speeds:
        subset = df[df['Speed'] == v].groupby('MPE')['Correct'].mean().reset_index()
        try:
            popt, _ = curve_fit(logistic_func, subset['MPE'], subset['Correct'],
                                p0=[3.0, 1.0], bounds=([0.1, 0.1], [20.0, 5.0]))
            results[v] = {'alpha': popt[0], 'beta': popt[1]}
        except Exception as e:
            print(f"  fit failed for speed {v}: {e}")
            results[v] = {'alpha': np.nan, 'beta': np.nan}
    return results

def aggregate_by_condition(df):
    # LMM -- supervisor want 3 degradation x 3 speeds
    df_agg = df.groupby(['PID', 'Speed', 'MPE']).agg(
        Accuracy=('Correct', 'mean'),
        RT_mean=('RT', 'mean'),
        N_trials=('Correct', 'count')
    ).reset_index()
    return df_agg

def compute_mi_from_accuracy(df_agg):
    # accuracy-based MI instead of threshold-based
    pid_static = df_agg[df_agg['Speed'] == 0].groupby('PID')['Accuracy'].mean()
    mi_data = {}
    for v in [1.5, 5.0, 15.0]:
        pid_dyn = df_agg[df_agg['Speed'] == v].groupby('PID')['Accuracy'].mean()
        mi_data[v] = ((pid_static - pid_dyn) / pid_static).values
    return mi_data
