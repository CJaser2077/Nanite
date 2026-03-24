import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from psych_fitting import logistic_func

def plot_all(df, df_agg, global_thresholds, mi_data, model_params, nanite_stats=None):
    sns.set_theme(style="whitegrid")
    outdir = '.'

    #Fig 5.1 psychometric curves
    plt.figure(figsize=(8, 5))
    x_range = np.linspace(0.5, 10, 200)
    colors = {'0.0': '#1f77b4', '1.5': '#ff7f0e', '5.0': '#2ca02c', '15.0': '#d62728'}
    labels = {'0.0': 'Static (0 m/s)', '1.5': '1.5 m/s', '5.0': '5.0 m/s', '15.0': '15.0 m/s'}

    for v in [0.0, 1.5, 5.0, 15.0]:
        p = global_thresholds[v]
        if np.isnan(p['alpha']):
            continue
        y = logistic_func(x_range, p['alpha'], p['beta'])
        plt.plot(x_range, y, label=labels[str(v)], color=colors[str(v)], lw=2)
        # mark q75 point
        plt.plot(p['alpha'], 0.75, 'o', color=colors[str(v)], ms=8, zorder=5)

    for v in [0.0, 1.5, 5.0, 15.0]:
        pts = df[df['Speed'] == v].groupby('MPE')['Correct'].mean()
        plt.scatter(pts.index, pts.values, color=colors[str(v)], s=40, zorder=6, edgecolors='white')

    plt.axhline(0.75, color='grey', ls='--', alpha=0.4, label='75% threshold')
    plt.axhline(0.50, color='grey', ls=':', alpha=0.3)
    plt.xlabel('Nanite Degradation Level (MPE)')
    plt.ylabel('Detection Accuracy')
    plt.title('Fig 5.1: Psychometric Functions across Speed Conditions')
    plt.legend(loc='lower right')
    plt.ylim(0.4, 1.05)
    plt.tight_layout()
    plt.savefig(f'{outdir}/Figure_5_1.png', dpi=300)
    plt.close()

    # Fig 5.2 MI bar chart
    plt.figure(figsize=(7, 5))
    speed_labels = ['1.5 m/s', '5.0 m/s', '15.0 m/s']
    mi_means = [np.nanmean(mi_data[v]) * 100 for v in [1.5, 5.0, 15.0]]
    mi_sds = [np.nanstd(mi_data[v]) * 100 for v in [1.5, 5.0, 15.0]]

    bars = plt.bar(speed_labels, mi_means, yerr=mi_sds, capsize=5,
                   color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8, edgecolor='black', lw=0.5)

    for i, v in enumerate([1.5, 5.0, 15.0]):
        vals = mi_data[v][~np.isnan(mi_data[v])] * 100
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        plt.scatter([i + j for j in jitter], vals, color='black', alpha=0.35, s=20, zorder=5)

    plt.ylabel('Motion Masking Index (%)')
    plt.title('Fig 5.2: Motion Masking Effect by Speed')
    plt.tight_layout()
    plt.savefig(f'{outdir}/Figure_5_2.png', dpi=300)
    plt.close()

    #Fig 5.3 speed-degradation model
    fig, ax1 = plt.subplots(figsize=(8, 5))
    v_pts = np.array([0.0, 1.5, 5.0, 15.0])
    q_pts = np.array([global_thresholds[v]['alpha'] for v in v_pts])

    ax1.scatter(v_pts, q_pts, color='darkblue', s=60, zorder=5, label='Empirical q75')
    v_smooth = np.linspace(0, 16, 200)
    q_fit = model_params[0] * np.log1p(v_smooth) + model_params[1]
    ax1.plot(v_smooth, q_fit, 'b--', alpha=0.7, lw=1.5, label='Log fit')
    ax1.set_xlabel('Movement Speed (m/s)')
    ax1.set_ylabel('Perceptual Threshold (MPE)', color='blue')

    if nanite_stats is not None:
        ax2 = ax1.twinx()
        ax2.plot(nanite_stats['MPE'], nanite_stats['Ratio'] * 100, 'rs-', lw=1.5, label='Triangle Ratio')
        ax2.set_ylabel('Triangle Load (%)', color='red')

    ax1.legend(loc='upper center')
    plt.title('Fig 5.3: Speed-Degradation Function')
    fig.tight_layout()
    plt.savefig(f'{outdir}/Figure_5_3.png', dpi=300)
    plt.close()

    #Fig 5.4 RT by speed
    plt.figure(figsize=(7, 5))
    rt_by_speed = df.groupby('Speed')['RT'].agg(['mean', 'sem']).reset_index()
    plt.errorbar(rt_by_speed['Speed'], rt_by_speed['mean'], yerr=rt_by_speed['sem'],
                 fmt='o-', color='green', capsize=5, lw=1.5, ms=7)
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Response Time (s)')
    plt.title('Fig 5.4: Mean Reaction Time across Speeds')
    plt.tight_layout()
    plt.savefig(f'{outdir}/Figure_5_4.png', dpi=300)
    plt.close()

    #Fig 5.5: accuracy heatmap (Speed * MPE)
    plt.figure(figsize=(6, 4))
    acc_pivot = df_agg.groupby(['Speed', 'MPE'])['Accuracy'].mean().unstack()
    sns.heatmap(acc_pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
                linewidths=0.5, cbar_kws={'label': 'Accuracy'})
    plt.title('Fig 5.5: Accuracy by Speed x MPE')
    plt.ylabel('Speed (m/s)')
    plt.xlabel('MPE Level')
    plt.tight_layout()
    plt.savefig(f'{outdir}/Figure_5_5.png', dpi=300)
    plt.close()
    print("  all figures saved")
