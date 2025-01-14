# SPDX-FileCopyrightText: 2021 Sebastian Garcia <sebastian.garcia@agents.fel.cvut.cz>
#  SPDX-License-Identifier: GPL-2.0-only
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import argparse
import datetime
from pathlib import Path

from common.utils import get_threshold
from common.models import get_triplet_loss_model
from common.data_loading import load_mal_data, load_ben_data

data_dir = Path("../data")

# tf.config.run_functions_eagerly(True)


def create_dataset(day, num_clients=10):
    clients = np.random.choice(range(1, 11), num_clients, replace=False)
    print(clients)
    data = dict()
    # for j in range(1, 6):
    data = pd.DataFrame()
    for i in clients:
        df_temp, _ = load_ben_data(day, i, data_dir)
        data = pd.concat([data, df_temp], ignore_index=True)

    mal_data = load_mal_data(1, data_dir)

    neg_df = load_mal_data(day, data_dir)["CTU-Malware-Capture-Botnet-67-1"]

    print("Size of negative dataframe:", len(neg_df))

    num_samples = len(data)

    idx = np.random.randint(0, num_samples, 15)
    pos_samples = data.values[idx]
    pos_samples = np.resize(pos_samples, (num_samples, 36))
    np.random.shuffle(pos_samples)

    neg_samples = np.resize(neg_df.values, (num_samples, 36))
    np.random.shuffle(neg_samples)

    print(pos_samples.shape, neg_samples.shape)

    return data, mal_data, pos_samples[:num_samples], neg_samples[:num_samples]


def metrics(fp, tp, num_ben, num_malware):
    # fp = anomalies_ben
    # tp = anomalies_mal
    tn = num_ben - fp
    # fn = num_malware - tp

    accuracy = (tp + tn) / (num_ben + num_malware)
    tpr = tp / num_malware
    fpr = fp / num_ben

    return accuracy, tpr, fpr


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Centralized training scenario 1a")
    parser.add_argument(
        "--day",
        type=int,
        choices=(range(1, 6)),
        required=True,
        help="Day to use for training.",
    )
    parser.add_argument(
        "--seed", type=int, required=False, default=8181, help="Random state seed."
    )
    parser.add_argument(
        "--num_clients",
        "-n",
        type=int,
        choices=(range(1, 11)),
        required=False,
        default=10,
    )
    args = parser.parse_args()

    day = args.day
    num_clients = args.num_clients

    tf.keras.utils.set_random_seed(args.seed)
    model = get_triplet_loss_model()
    data, mal_data, X_pos, X_neg = create_dataset(day, num_clients)
    mal_folders = list(mal_data.keys())

    EPOCHS = 10
    BATCH_SIZE = 64

    scaler = MinMaxScaler()
    X = scaler.fit_transform(data)
    X_pos = scaler.transform(X_pos)
    X_neg = scaler.transform(X_neg)

    print(X.shape)
    X_test, _, _, _ = create_dataset(day + 1, 10)
    X_test = scaler.transform(X_test)

    X_train, X_val = train_test_split(X, test_size=0.2, random_state=args.seed)
    num_train = X_train.shape[0]
    # num_val = X_val.shape[0]

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    model.fit(
        x=(X_train, X_pos[:num_train], X_neg[:num_train]),
        y=X_train,
        shuffle=True,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=((X_val, X_pos[num_train:], X_neg[num_train:]), X_val),
        callbacks=[tensorboard_callback],
    )

    rec_ben = model.predict((X_test, X_test, X_test))
    mse_ben = np.mean(np.power(X_test - rec_ben, 2), axis=1)

    rec_mal = dict()
    mse_mal = dict()
    num_malware = 0
    for folder in mal_folders:
        X_test_mal = scaler.transform(mal_data[folder])
        num_malware += X_test_mal.shape[0]
        rec_mal[folder] = model.predict((X_test_mal, X_test_mal, X_test_mal))
        mse_mal[folder] = np.mean(np.power(X_test_mal - rec_mal[folder], 2), axis=1)

    # The number of faulty samples for a 2% FPR (on the training set)
    rec_val = model.predict((X_val, X_pos[num_train:], X_neg[num_train:]))
    mse_val = np.mean(np.power(X_val - rec_val, 2), axis=1)
    th = get_threshold(X_val, mse_val, 0.01)
    print(f"Calculated threshold: {th:.5f}")

    # Measure in the testset
    rec_ben = model.predict((X_test, X_test, X_test))
    mse_ben = np.mean(np.power(X_test - rec_ben, 2), axis=1)
    print(
        f"False positives on next day: { 100*sum(mse_ben > th) / len(X_test):.2f}% ({sum(mse_ben > th)} out of {len(X_test)})"
    )
    anomalies_ben = sum(mse_ben > th)
    num_examples_test = X_test.shape[0]

    anomalies_mal = 0
    for folder in mal_folders:
        anomalies_mal += sum(mse_mal[folder] > th)
        print(
            f"{folder} detected: {100*sum(mse_mal[folder] > th) / len(mse_mal[folder]):.2f}% ({sum(mse_mal[folder] > th)} out of {len(mse_mal[folder])})"
        )

    accuracy, tpr, fpr = metrics(
        anomalies_ben, anomalies_mal, num_examples_test, num_malware
    )

    # Metrics on the test set for both malware and benign data
    print(f"Centralized accuracy: {100*accuracy:.2f}%")
    print(f"Centralized tpr: {100*tpr:.2f}%")
    print(f"Centralized fpr: {100*fpr:.2f}%")


if __name__ == "__main__":
    main()
