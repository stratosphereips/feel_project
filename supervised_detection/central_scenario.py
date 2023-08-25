import pickle
from copy import deepcopy

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import argparse
import os
import datetime
from matplotlib import pyplot as plt

from common.config import Config
from common.models import get_classification_model, MultiHeadAutoEncoder, SimpleClassifier
from common.data_loading import load_all_data, load_centralized_data
from common.utils import MinMaxScaler, pprint_cm, get_threshold, plot_embedding
from pathlib import Path
import fire

import umap

data_dir = Path("../data")

# tf.config.run_functions_eagerly(True)


def create_dataset(day, config: Config):
    return load_centralized_data(day, config)


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
    day = day

    tf.keras.utils.set_random_seed(config.seed)
    model = SimpleClassifier(config)
    model.compile(metrics=['BinaryCrossentropy', 'AUC', 'Accuracy'])
    if config.load_model and day > 1 and config.model_file(day - 1).exists():
        model.load_weights(config.model_file(day - 1))
        epochs = config.server.num_rounds_other_days
    else:
        epochs = config.server.num_rounds_first_day

    # get_classification_model(Path(model_path), 2, encoder_lr=1e-4, classifier_lr=1e-4)
    X_train, X_val, X_test, y_train, y_val, y_test = create_dataset(day, config)

    EPOCHS = sum(config.local_epochs(round) for round in range(1, epochs + 1))
    BATCH_SIZE = config.client.batch_size

    scaler = MinMaxScaler().fit(X_train)
    X_train, X_val, X_test = (
        scaler.transform(X_train),
        scaler.transform(X_val),
        scaler.transform(X_test),
    )

    X_train = X_train.astype(
        "float32"
    )
    X_val = X_val.astype("float32")
    X_test = X_test

    logs_dir = Path(f"../logs/fit/centralized") / datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logs_dir, histogram_freq=1
    )

    eval_callback = EvaluateCallback(
        model, X_test, y_test, X_val, y_val
    )

    history = model.fit(
        x=X_train,
        y=y_train,
        shuffle=True,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard_callback, eval_callback],
    )

    model.set_weights(eval_callback.best_weights)

    y_pred_raw = model.predict(X_test)

    print(y_pred_raw)
    y_pred = (y_pred_raw[:, 0] > 0.5).astype(float).T

    report = classification_report(
        y_test, y_pred, target_names=["Benign", "Malicious"], output_dict=True
    )
    conf_matrix = confusion_matrix(y_test, y_pred)

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
    print(f"Centralized classification accuracy: {100*report['accuracy']:.2f}%")
    print(f"Centralized classification tpr: {100*tpr:.2f}%")
    print(f"Centralized classification fpr: {100*fpr:.2f}%")

    pprint_cm(conf_matrix, ["Benign", "Malicious"])

    # saving model and results
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
        total_loss = logs["val_loss"]
        if total_loss <= self.best_loss:
            self.best_loss = total_loss
            self.best_weights = deepcopy(self.model.get_weights())

        num_malicious = self.y_test.sum()

        y_pred_raw = self.model.predict(self.X_test)
        y_pred = (y_pred_raw[:, -1] > 0.5).astype(float).T

        cls_acc = (y_pred == self.y_test).mean()

        num_malware = (self.y_test == 1).sum()
        num_benign = (self.y_test == 0).sum()

        cls_mal_mask = y_pred == 1
        cls_mal_true = self.y_test[cls_mal_mask]

        cls_fp = int((cls_mal_true == 0).sum())
        cls_tp = int((cls_mal_true == 1).sum())

        cls_ben_true = self.y_test[y_pred == 0]
        cls_tn = int((cls_ben_true == 0).sum())
        cls_fn = int((cls_ben_true == 1).sum())

        cls_tpr = (cls_tp / num_malware) if num_malicious else np.nan
        cls_fpr = cls_fp / num_benign

        eval_results = {
            "cls_fp": cls_fp,
            "cls_tp": cls_tp,
            "cls_tn": cls_tn,
            "cls_fn": cls_fn,
            "cls_tpr": cls_tpr,
            "cls_fpr": cls_fpr,
            "class_accuracy": cls_acc
        }
        for key, value in eval_results.items():
            logs[key] = value
        return logs


if __name__ == "__main__":
    fire.Fire(main)
