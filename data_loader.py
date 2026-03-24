import pandas as pd

def load_and_preprocess(file_path):
    import pandas as pd
    df_raw = pd.read_csv(file_path)
    raw_count = len(df_raw)
    # filter out accidental clicks and distracted trials
    df_cleaned = df_raw[(df_raw['RT'] >= 0.2) & (df_raw['RT'] <= 3.0)].copy()
    clean_count = len(df_cleaned)
    removed_count = raw_count - clean_count
    print(f"  Raw trials loaded:  {raw_count}")
    print(f"  Outliers removed:   {removed_count} (RT < 0.2s or > 3.0s)")
    print(f"  Valid trials:       {clean_count}")
    print(f"  Retention rate:     {(clean_count / raw_count) * 100:.2f}%")
    return df_cleaned

def load_subjective_data(file_path):
    return pd.read_csv(file_path)
def load_nanite_stats(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"  [warn] {file_path} not found!!!")
        return None
