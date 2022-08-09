import pandas as pd
import numpy as np
import tensorflow as tf
import os

data_dir = '/opt/Malware-Project/BigDataset/FEELScenarios/'
# TIME_STEPS = 288

# # Generated training sequences for use in the model.
# def create_sequences(values, time_steps=TIME_STEPS):
#     output = []
#     for i in range(len(values) - time_steps + 1):
#         output.append(values[i : (i + time_steps)])
#     return np.stack(output)

# def get_data():
    
#     master_url_root = "https://raw.githubusercontent.com/numenta/NAB/master/data/"

#     df_small_noise_url_suffix = "artificialNoAnomaly/art_daily_small_noise.csv"
#     df_small_noise_url = master_url_root + df_small_noise_url_suffix
#     df_small_noise = pd.read_csv(
#         df_small_noise_url, parse_dates=True, index_col="timestamp"
#     )

#     df_daily_jumpsup_url_suffix = "artificialWithAnomaly/art_daily_jumpsup.csv"
#     df_daily_jumpsup_url = master_url_root + df_daily_jumpsup_url_suffix
#     df_daily_jumpsup = pd.read_csv(
#         df_daily_jumpsup_url, parse_dates=True, index_col="timestamp"
#     )

#     training_mean = df_small_noise.mean()
#     training_std = df_small_noise.std()
#     df_training_value = (df_small_noise - training_mean) / training_std


#     x_train = create_sequences(df_training_value.values)

#     # Prepare test data
#     df_test_value = (df_daily_jumpsup - training_mean) / training_std

#     # Create sequences from test values.
#     x_test = create_sequences(df_test_value.values)

#     return x_train, x_test

# def get_model():
#     model = tf.keras.Sequential(
#     [
#         tf.keras.layers.Input(shape=(TIME_STEPS, 1)),
#         tf.keras.layers.Conv1D(
#             filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
#         ),
#         tf.keras.layers.Dropout(rate=0.2),
#         tf.keras.layers.Conv1D(
#             filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
#         ),
#         tf.keras.layers.Conv1DTranspose(
#             filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
#         ),
#         tf.keras.layers.Dropout(rate=0.2),
#         tf.keras.layers.Conv1DTranspose(
#             filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
#         ),
#         tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
#     ]
#     )
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

#     return model


def get_model():
    model = tf.keras.Sequential(
        [
        tf.keras.layers.Input(shape=(40)),
        tf.keras.layers.Dense(36, activation='elu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(20, activation='elu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='elu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(20, activation='elu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(36, activation='elu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(40, activation='elu')
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def get_ben_data(day: int, client_id: int):
    
    df_ben = df_temp = pd.read_csv(os.path.join(data_dir, 'Processed', 'Client'+str(client_id), 'Day'+str(day), "comb_features_ben.csv"))
    df_ben_test = df_temp = pd.read_csv(os.path.join(data_dir, 'Processed', 'Client'+str(client_id), 'Day'+str(day+1), "comb_features_ben.csv"))

    df_ben = df_ben.drop(["ssl_ratio"], axis=1)
    df_ben = df_ben.drop_duplicates()

    df_ben_test = df_ben_test.drop(["ssl_ratio"], axis=1)
    df_ben_test = df_ben_test.drop_duplicates()

    return df_ben, df_ben_test


def get_mal_data():
    mal_data = dict()
    mal_folders = ['CTU-Malware-Capture-Botnet-346-1', 'CTU-Malware-Capture-Botnet-327-2', 'CTU-Malware-Capture-Botnet-230-1', 'CTU-Malware-Capture-Botnet-219-2']

    for folder in mal_folders:
        mal_data[folder] = pd.DataFrame()
        df_temp = pd.read_csv(os.path.join(data_dir, 'Raw', 'Malware', folder, 'Day1', "comb_features.csv"))
        mal_data[folder] = pd.concat([mal_data[folder], df_temp], ignore_index=True)

    for folder in mal_folders:
        mal_data[folder] = mal_data[folder].drop(["ssl_ratio"], axis=1)
        mal_data[folder] = mal_data[folder].drop_duplicates()  
         
    return mal_data

def get_threshold(X, mse):
    num = 0.02*len(X)

    th = 0.001
    while (sum(mse > th) > num):
        th += 0.001
    return th