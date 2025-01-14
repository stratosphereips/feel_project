# SPDX-FileCopyrightText: 2021 Sebastian Garcia <sebastian.garcia@agents.fel.cvut.cz>
#  SPDX-License-Identifier: GPL-2.0-only
import pickle
from copy import deepcopy
from pydoc import cli
import numpy as np
import tensorflow as tf
import pandas as pd
from fire import Fire
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path

from common.config import Config
from common.utils import get_threshold
from common.models import get_ad_model, MultiHeadAutoEncoder
from common.data_loading import load_mal_data, load_ben_data, load_centralized_data

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

data_dir = Path("../data")


def create_dataset(day, config: Config):
    return load_centralized_data(day, config, train_malicious=False)


def metrics(fp, tp, num_ben, num_malware):
    # fp = anomalies_ben
    # tp = anomalies_mal
    tn = num_ben - fp
    # fn = num_malware - tp

    accuracy = (tp + tn) / (num_ben + num_malware)
    tpr = tp / num_malware
    fpr = fp / num_ben

    return accuracy, tpr, fpr


def main(day: int, config=None, **overrides):
    config = Config.load(config, **overrides)

    num_clients = config.num_fit_clients

    tf.keras.utils.set_random_seed(config.seed)
    model = MultiHeadAutoEncoder(config)
    model.compile()

    X_train, X_val, X_test, y_train, y_val, y_test = create_dataset(day, config)

    if config.load_model and day > 1 and config.model_file(day - 1).exists():
        model.load_weights(config.model_file(day - 1))
        EPOCHS = config.server.num_rounds_other_days
    else:
        EPOCHS = config.server.num_rounds_first_day

    EPOCHS = sum(config.local_epochs(round) for round in range(1, EPOCHS + 1))

    BATCH_SIZE = config.client.batch_size

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train, X_val, X_test = (
        scaler.transform(X_train),
        scaler.transform(X_val),
        scaler.transform(X_test),
    )

    X_train_l = np.hstack([X_train, np.zeros((X_train.shape[0], 1))])
    X_val_l = np.hstack([X_train, np.zeros((X_train.shape[0], 1))])

    eval_callback = EvaluateCallback(model, X_test, y_test, X_val, y_val)

    history = model.fit(
        X_train_l,
        X_train_l,
        shuffle=True,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_l, X_val_l),
        callbacks=[eval_callback],
    )

    model.set_weights(eval_callback.best_weights)

    # The number of faulty samples for a 2% FPR (on the training set)
    rec_val = model.predict(X_val)[:, :-1]
    mse_val = np.mean(np.power(X_val - rec_val, 2), axis=1)
    th = get_threshold(X_val, mse_val)

    print(f"Calculated threshold: {th:.5f}")

    # Measure in the testset
    rec = model.predict(X_test)[:, :-1]
    mse_ben = np.mean(np.power(X_test - rec, 2), axis=1)
    y_pred = (rec[:, -1] > th).astype(float).T

    report = classification_report(
        y_test, y_pred, target_names=["Benign", "Malicious"], output_dict=True
    )
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = (y_pred == y_test).mean()

    num_malware = (y_test == 1).sum()
    num_benign = (y_test == 0).sum()

    anomalies_mask = y_pred == 1
    anomalies_true = y_test[anomalies_mask]

    fp = int((anomalies_true == 0).sum())
    tp = int((anomalies_true == 1).sum())
    tn = len(y_test) - fp

    tpr = tp / num_malware
    fpr = fp / num_benign

    # Metrics on the test set for both malware and benign data
    print(f"Centralized accuracy: {100*accuracy:.2f}%")
    print(f"Centralized tpr: {100*tpr:.2f}%")
    print(f"Centralized fpr: {100*fpr:.2f}%")
    model.save_weights(config.model_file(day))
    with config.results_file(day).open("wb") as f:
        pickle.dump(history.history, f)
    with config.scaler_file(day).open("wb") as f:
        pickle.dump(scaler, f)


class EvaluateCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, X_test, y_test, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.best_weights = None
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        total_loss = logs["val_total_loss"]
        if total_loss <= self.best_loss:
            self.best_loss = total_loss
            self.best_weights = deepcopy(self.model.get_weights())

        ben_val = self.X_val[(self.y_val == 0)]
        rec = self.model.predict(ben_val)[:, :-1]
        mse = np.mean(np.power(ben_val - rec, 2), axis=1)
        threshold = get_threshold(ben_val, mse)

        num_malicious = self.y_test.sum()

        num_malware = (self.y_test == 1).sum()
        num_benign = (self.y_test == 0).sum()

        rec = self.model.predict(self.X_test.astype("float32"))[:, :-1]
        mse = np.mean(np.power(self.X_test - rec, 2), axis=1)
        y_anomaly_pred = (mse > threshold).astype("float32")

        anomalies_mask = y_anomaly_pred == 1
        anomalies_true = self.y_test[anomalies_mask]

        ad_fp = int((anomalies_true == 0).sum())
        ad_tp = int((anomalies_true == 1).sum())

        anomalies_false = self.y_test[y_anomaly_pred == 0]
        ad_tn = int((anomalies_false == 0).sum())
        ad_fn = int((anomalies_false == 1).sum())

        ad_tpr = (ad_tp / num_malware) if num_malicious else np.nan
        ad_fpr = ad_fp / num_benign

        ad_accuracy = (y_anomaly_pred == self.y_test).mean()
        eval_results = {
            "fp": ad_fp,
            "tp": ad_tp,
            "tn": ad_tn,
            "fn": ad_fn,
            "tpr": ad_tpr,
            "fpr": ad_fpr,
            "acc": ad_accuracy,
        }
        for key, value in eval_results.items():
            logs[key] = value
        return logs


if __name__ == "__main__":
    Fire(main)
