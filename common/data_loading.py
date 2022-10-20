from pathlib import Path
import pandas as pd


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
    mal_data = dict()
    mal_dir = data_dir / 'Processed' / 'Malware'
    for folder in mal_dir.iterdir():
        mal_data[folder.name] = _load_data(folder / f'Day{day}', drop_labels, drop_four_tuple)

    return mal_data


def _load_data(data_dir: Path, drop_labels, drop_four_tuple, drop_malicious=False):
    df = pd.read_csv(data_dir / 'comb_features.csv')

    dropped_cols = ["ssl_ratio", "self_signed_ratio", "SNI_equal_DstIP", "ratio_certificate_path_error",
                    "ratio_missing_cert_in_cert_path"]
    if drop_labels:
        dropped_cols += ['label', 'detailedlabel']
    if drop_four_tuple:
        dropped_cols += ['id.orig_h', 'id.resp_h', 'id.orig_p', 'proto']

    if drop_malicious:
        df = df[df.label != 'Malicious']

    return df.drop(dropped_cols, axis=1) \
        .drop_duplicates()
