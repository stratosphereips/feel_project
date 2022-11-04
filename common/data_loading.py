from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import defaultdict
import numpy as np

_label_cols = ['label', 'detailedlabel']
_id_cols = ['id.orig_h', 'id.resp_h', 'id.orig_p', 'proto']


def load_ben_data(day: int, client_id: int, data_dir: Path, drop_labels=True, drop_four_tuple=True):
    client_dir = data_dir / 'Processed' / f'Client{client_id}'

    df_ben = _load_data(client_dir / f'Day{day}', drop_labels, drop_four_tuple, drop_malicious=True)
    df_ben_test = _load_data(
        client_dir / f'Day{day + 1}',
        drop_labels,
        drop_four_tuple,
        drop_malicious=True
    ) if day < 5 else pd.DataFrame(columns=df_ben.columns)

    return df_ben, df_ben_test


def load_mal_data(day: int, data_dir: Path, drop_labels=True, drop_four_tuple=True):
    mal_data = defaultdict(pd.DataFrame)
    mal_dir = data_dir / 'Processed' / 'Malware'
    for folder in mal_dir.iterdir():
        mal_data[folder.name] = _load_data(folder / f'Day{day}', drop_labels, drop_four_tuple)

    return mal_data


def create_supervised_dataset(df_ben: pd.DataFrame, df_mal: pd.DataFrame, test_ratio=0.2, seed=None, resize_mal=False):
    if resize_mal:
        resized_array = np.resize(df_mal.to_numpy(), (df_ben.shape[0], df_mal.shape[1]))
        df_mal = pd.DataFrame(resized_array, columns=df_mal.columns)

    df = pd.concat((df_ben, df_mal), axis=0)

    y = (df.label == 'Malicious').astype('double').to_numpy()
    X = df.drop(_label_cols, axis=1).to_numpy()

    if not test_ratio:
        X, y = shuffle(X, y, random_state=seed)
        return X, None, y, None
    else:
        return train_test_split(X, y, test_size=test_ratio, shuffle=True, random_state=seed)


def load_all_data(day: int, data_dir: Path, drop_labels=True, drop_four_tuple=True, seed=None):
    ben_train, ben_test = zip(
        *[load_ben_data(day, client, data_dir, drop_labels=False, drop_four_tuple=drop_four_tuple) for client in range(1, 10)])
    X_ben_train = pd.concat(ben_train, axis=0)
    X_ben_test = pd.concat(ben_test, axis=0)

    X_mal_train = pd.concat(load_mal_data(day, data_dir, drop_labels=False, drop_four_tuple=drop_four_tuple).values(), axis=0)
    X_mal_test = pd.concat(load_mal_data(day + 1, data_dir, drop_labels=False, drop_four_tuple=drop_four_tuple).values(), axis=0)

    X_train, X_val, y_train, y_val = create_supervised_dataset(X_ben_train, X_mal_train, 0.2, seed)
    X_test, _, y_test, _ = create_supervised_dataset(X_ben_test, X_mal_test, 0.0, seed)

    return X_train, X_val, X_test, y_train, y_val, y_test


def _load_data(data_dir: Path, drop_labels, drop_four_tuple, drop_malicious=False):
    features_file = data_dir / 'comb_features.csv'
    if not features_file.exists():
        return pd.DataFrame()
    df = pd.read_csv(features_file)

    dropped_cols = ["ssl_ratio", "self_signed_ratio", "SNI_equal_DstIP", "ratio_certificate_path_error",
                    "ratio_missing_cert_in_cert_path"]
    if drop_labels:
        dropped_cols += _label_cols
    if drop_four_tuple:
        dropped_cols += _id_cols

    if drop_malicious:
        df = df[df.label == 'Benign']

    return df.drop(dropped_cols, axis=1) \
        .drop_duplicates()
