
import pandas as pd

import warnings

warnings.filterwarnings("ignore", message=".*had to be resampled from.*")
warnings.filterwarnings("ignore", message="Warning: input samples dtype is np.float64. Converting to np.float32")
warnings.filterwarnings("ignore", message="Xing stream size off by more than 1%")

def create_data_rare_bird(df_path, min_count=35):
    df = pd.read_parquet(df_path)

    rare_label = df.primary_label.value_counts()[df.primary_label.value_counts() <= min_count].index
    df_rare = df[df.primary_label.isin(rare_label)]
    df_other = df[~df.primary_label.isin(rare_label)]

    labels = df_rare['primary_label'].unique()
    dictionary = {label: i for i, label in enumerate(labels)}
    print(len(dictionary))
    print(dictionary)

    df_rare['rare_label'] = df_rare['primary_label'].map(dictionary).astype(int)
    df_other['rare_label'] = -1

    df = pd.concat([df_rare, df_other]).reset_index()

    new_features = []
    len_sec = []
    for i, row in df.iterrows():
        row_lst = []
        if row["count_sec_labels"] == 0:
            new_features.append([])
        else:
            for v in row["secondary_labels"]:
                if v in dictionary:
                    row_lst.append(dictionary[v])
            new_features.append(row_lst)
        len_sec.append(len(row_lst))

    df['rare_sec_label_int'] = new_features
    df['count_sec_rare'] = len_sec

    return df