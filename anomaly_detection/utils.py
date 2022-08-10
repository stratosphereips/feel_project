import pandas as pd
import numpy as np
import tensorflow as tf
import os

data_dir = '/opt/Malware-Project/BigDataset/FEELScenarios/'


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
    
    df_ben = pd.read_csv(os.path.join(data_dir, 'Processed', 'Client'+str(client_id), 'Day'+str(day), "comb_features_ben.csv"))
    df_ben_test = pd.read_csv(os.path.join(data_dir, 'Processed', 'Client'+str(client_id), 'Day'+str(day+1), "comb_features_ben.csv"))

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
    num = 0.01*len(X)

    th = 0.001
    while (sum(mse > th) > num):
        th += 0.001
    return th


def mad_threshold(mse):
    # return 3.5*np.median(np.abs(mse - np.median(mse)))
    return 3.5*np.median(np.absolute(mse - np.median(mse)))


def mad_score(points):
    """https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm """
    m = np.median(points)
    ad = np.abs(points - m)
    mad = np.median(ad)
    
    return 0.6745 * ad / mad

