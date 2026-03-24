import pandas as pd
import numpy as np

# generate performance projection table from the fitted model params
# supervisor: figure out nanite triangles for meshes at each level and report findings
def generate_projection_table(nanite_stats_file, a, b):
    try:
        stats = pd.read_csv(nanite_stats_file)
    except FileNotFoundError:
        print(f"  cant find {nanite_stats_file}!!!")
        return

    print("\n nanite triangle stats per MPE level")
    print(stats.to_string(index=False))
    speeds = [0, 1.5, 5.0, 10.0, 15.0]
    projection_data = []

    for v in speeds:
        q75_theory = a * np.log1p(v) + b
        q75_theory = max(1.0, q75_theory)
        available_mpe = stats['MPE'].values
        safe_mpe_list = available_mpe[available_mpe <= q75_theory]
        best_mpe = max(safe_mpe_list) if len(safe_mpe_list) > 0 else 1
        row = stats[stats['MPE'] == best_mpe].iloc[0]
        tri_ratio = row['Ratio']
        savings = (1 - tri_ratio) * 100
        projection_data.append({
            'Speed (m/s)': v,
            'Threshold (q75)': round(q75_theory, 2),
            'Recommended MPE': int(best_mpe),
            'Triangle Load (%)': f'{tri_ratio * 100:.2f}%',
            'Saving (%)': f'{savings:.2f}%'
        })
    result_df = pd.DataFrame(projection_data)
    result_df.to_csv('Performance_Projection_Table.csv', index=False)
    print(result_df.to_string(index=False))

