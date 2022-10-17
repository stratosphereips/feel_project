import pandas as pd
import numpy as np
import tensorflow as tf
import os
from pathlib import Path
import base64

class Encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim=10, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.l1 = tf.keras.layers.Dense(32, activation='elu')
        self.l2 = tf.keras.layers.Dense(20, activation='elu')
        self.l3 = tf.keras.layers.Dense(latent_dim, activation='sigmoid')
        self.drop = tf.keras.layers.Dropout(0.3)

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.drop(x)
        x = self.l2(x)
        x = self.drop(x)
        z = self.l3(x)

        return z


class Decoder(tf.keras.layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim=36, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.l1 = tf.keras.layers.Dense(20, activation='elu')
        self.l2 = tf.keras.layers.Dense(32, activation='elu')
        self.l3 = tf.keras.layers.Dense(original_dim, activation='elu')
        self.drop = tf.keras.layers.Dropout(0.3)

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.drop(x)
        x = self.l2(x)
        x = self.drop(x)
        return self.l3(x)

class AutoEncoder(tf.keras.layers.Layer):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        # self.original_dim = 36
        self.encoder = Encoder(latent_dim=10)
        self.decoder = Decoder(36)
        self.margin = 1.0

    def call(self, inputs, positive, negative):
        z = self.encoder(inputs)
        p = self.encoder(positive)
        n = self.encoder(negative)
        reconstructed = self.decoder(z)

        # Add Triplet loss.
        # The triplet loss is defined as:
        # L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)

        ap_distance = tf.reduce_sum(tf.square(z - p))
        an_distance = tf.reduce_sum(tf.square(z - n))
        
        triplet_loss = ap_distance - an_distance
        triplet_loss = tf.maximum(triplet_loss + self.margin, 0.0)
        # triplet_loss = (ap_distance + tf.maximum(0.0, (10. - an_distance)/10.)) / 2.

        self.add_loss(triplet_loss)
        # self.add_metric(triplet_loss, name="triplet_loss", aggregation="mean")

        return reconstructed


def get_model():
    inputs = tf.keras.Input(shape=(36))
    pos = tf.keras.Input(shape=(36))
    neg = tf.keras.Input(shape=(36))

    ae = AutoEncoder()
    outputs = ae(inputs, pos, neg)

    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    model = tf.keras.Model([inputs, pos, neg], outputs)
    model.compile(optimizer=optimizer, loss=loss_fn)

    return model


def get_model_old():
    model = tf.keras.Sequential(
        [
        tf.keras.layers.Input(shape=(36)),
        tf.keras.layers.Dense(32, activation='elu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(20, activation='elu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='elu'),
        # tf.keras.layers.Dropout(0.3),
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
    client_dir = data_dir / f'Client{client_id}'

    df_ben = pd.read_csv(client_dir / f'Day{day}' / "comb_features.csv")
    df_ben_test = pd.read_csv(client_dir / f'Day{day+1}' / "comb_features.csv")

    df_ben = df_ben[df_ben['label'] != 'Malicious']
    df_ben_test = df_ben_test[df_ben_test['label'] != "Malicious"]

    df_ben = df_ben.drop(["ssl_ratio", "self_signed_ratio", "SNI_equal_DstIP", "ratio_certificate_path_error", "ratio_missing_cert_in_cert_path", "label", "detailedlabel"], axis=1)
    df_ben = df_ben.drop_duplicates()

    df_ben_test = df_ben_test.drop(["ssl_ratio", "self_signed_ratio", "SNI_equal_DstIP", "ratio_certificate_path_error", "ratio_missing_cert_in_cert_path", "label", "detailedlabel"], axis=1)
    df_ben_test = df_ben_test.drop_duplicates()
    return df_ben, df_ben_test


def get_mal_data(data_dir: Path):
    mal_data = dict()
    mal_folders = ['CTU-Malware-Capture-Botnet-346-1', 'CTU-Malware-Capture-Botnet-327-2', 'CTU-Malware-Capture-Botnet-230-1', 'CTU-Malware-Capture-Botnet-219-2']
    mal_dir = data_dir / 'Malware'
    for folder in mal_folders:
        mal_data[folder] = pd.DataFrame()
        df_temp = pd.read_csv(mal_dir / folder / 'Day1' / "comb_features.csv")
        mal_data[folder] = pd.concat([mal_data[folder], df_temp], ignore_index=True)

    for folder in mal_folders:
        mal_data[folder] = mal_data[folder].drop(["ssl_ratio", "self_signed_ratio", "SNI_equal_DstIP", "ratio_certificate_path_error", "ratio_missing_cert_in_cert_path", "label", "detailedlabel"], axis=1)
        mal_data[folder] = mal_data[folder].drop_duplicates()

    return mal_data


def get_neg_data(day: int, client_id: int, data_dir: Path):
    neg_df = pd.read_csv((data_dir / f'Client{client_id}' / f'Day{day}' / "comb_features.csv"))
    neg_df = neg_df[neg_df['label'] == 'Malicious']
    neg_df = neg_df.drop(["ssl_ratio", "self_signed_ratio", "SNI_equal_DstIP", "ratio_certificate_path_error", "ratio_missing_cert_in_cert_path", "label", "detailedlabel"], axis=1)
    neg_df = neg_df.drop_duplicates()

    return neg_df

def get_threshold(X, mse, level=0.01):
    num = max(level*len(X), 2)

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