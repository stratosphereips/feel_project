import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import base64


def get_model():
    model = tf.keras.Sequential(
        [
        tf.keras.layers.Input(shape=(36)),
        tf.keras.layers.Dense(32, activation='elu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(20, activation='elu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='elu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(20, activation='elu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='elu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(36, activation='elu')
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def get_ben_data(day: int, client_id: int, data_dir: Path):
    
    df_ben = pd.read_csv(Path(data_dir / f'Client{client_id}' / f'Day{day}' / "comb_features.csv"))
    df_ben_test = pd.read_csv(Path(data_dir / f'Client{client_id}' / f'Day{day+1}' / "comb_features.csv"))

    df_ben = df_ben[df_ben.label != 'Malicious']
    df_ben_test = df_ben_test[df_ben_test.label != 'Malicious']

    df_ben = df_ben.drop(["ssl_ratio", "self_signed_ratio", "SNI_equal_DstIP", "ratio_certificate_path_error", "ratio_missing_cert_in_cert_path", 'label', 'detailedlabel'], axis=1)
    df_ben = df_ben.drop_duplicates()

    df_ben_test = df_ben_test.drop(["ssl_ratio", "self_signed_ratio", "SNI_equal_DstIP", "ratio_certificate_path_error", "ratio_missing_cert_in_cert_path", 'label', 'detailedlabel'], axis=1)
    df_ben_test = df_ben_test.drop_duplicates()

    return df_ben, df_ben_test


def get_mal_data(data_dir):
    mal_data = {}
    mal_folders = ['CTU-Malware-Capture-Botnet-346-1', 'CTU-Malware-Capture-Botnet-327-2', 'CTU-Malware-Capture-Botnet-230-1', 'CTU-Malware-Capture-Botnet-219-2']

    for folder in mal_folders:
        mal_data[folder] = pd.DataFrame()
        df_temp = pd.read_csv(Path(data_dir / 'Malware' / folder / 'Day1' / "comb_features.csv"))
        mal_data[folder] = pd.concat([mal_data[folder], df_temp], ignore_index=True)

    for folder, df in mal_data.items():
        df = df[df.label == 'Malicious']
        df = df.drop(["ssl_ratio", "self_signed_ratio", "SNI_equal_DstIP", "ratio_certificate_path_error", "ratio_missing_cert_in_cert_path", 'label', 'detailedlabel'], axis=1)
        # mal_data[folder] = mal_data[folder].drop(["ssl_ratio", "ratio_certificate_path_error", "ratio_missing_cert_in_cert_path"], axis=1)
        mal_data[folder] = df.drop_duplicates()

    return mal_data

def get_threshold(X, mse):
    num = max(0.01*len(X), 2)

    th = 0.0001
    while (sum(mse > th) > num):
        th += 0.0001
    return th

def serialize_array(arr):
    temp_str = ""
    for element in arr:
        temp_str += str(element)
        temp_str += '|'

    return base64.b64encode(bytes(temp_str, "utf-8"))

def deserialize_string(b64_str: str):
    arr = base64.b64decode(b64_str).decode("utf-8", "ignore")

    return [float(element) for element in arr.split('|') if element != '']

def scale_data(X, X_min, X_max):
    X_std = (X - X_min) / (X_max - X_min + 1e-5)
    X_std = np.nan_to_num(X_std, 0.5)
    return X_std