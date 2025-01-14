# SPDX-FileCopyrightText: 2021 Sebastian Garcia <sebastian.garcia@agents.fel.cvut.cz>
#  SPDX-License-Identifier: GPL-2.0-only
import datetime
import warnings

import pandas as pd
from flwr.common import parameters_to_ndarrays
from sklearn.model_selection import train_test_split

from common.config import Config, Setting

warnings.filterwarnings("ignore")

import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
from pathlib import Path

import flwr as fl
from flwr.common.logger import logger
from common.data_loading import (
    load_mal_data,
    load_ben_data,
    create_supervised_dataset,
    load_day_dataset,
    load_centralized_data,
)
from common.models import get_classification_model, MultiHeadAutoEncoder
from common.utils import (
    get_threshold,
    serialize_array,
    deserialize_string,
    client_malware_map,
    MinMaxScaler,
    serialize,
    deserialize,
)
from fire import Fire
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import logging

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class ADClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id,
        model,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        config,
        global_X_test=None,
        global_y_test=None,
    ):
        self.client_id = client_id
        self.log = logging.getLogger(f"Client {client_id}")
        self.model = model
        self.log_dir = Path(
            f"../logs/fit/client{self.client_id:02}"
        ) / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Store the data unscaled
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.has_own_malware = y_train.sum() > 0

        self.scaler = MinMaxScaler()
        self.scaler.fit(self.X_train)

        self.seed = config.seed
        self.proxy_radius_mult = config.client.proxy_radius_mult
        self.config = config
        self.threshold = 1000

        self.best_val_loss = np.inf
        self.best_val_params = model.get_weights()
        self.last_val_loss = np.nan
        self.val_losses = []
        self.val_acc = []

        self.global_X_test = global_X_test
        self.global_y_test = global_y_test

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        if not self.config.fit_if_no_malware and not self.has_own_malware:
            return [], 0, {}

        mal_dataset = deserialize(config["mal_dataset"])

        X_train, X_val, y_train, y_val = self.prepare_local_dataset(mal_dataset)
        X_train_l = np.concatenate([X_train, y_train.reshape((-1, 1))], axis=1).astype(
            "float32"
        )
        X_val_l = np.concatenate([X_val, y_val.reshape((-1, 1))], axis=1).astype(
            "float32"
        )

        num_examples_train = len(X_train_l)
        num_examples_mal = y_train.sum()

        if self.config.setting == Setting.FEDERATED:
            logger.info("Setting weights")
            self.model.set_weights(parameters)
        else:
            logger.info("Training locally only, ignoring new weights")

        # Get hyperparameters for this round
        start_epoch: int = config["start_epoch"]
        batch_size: int = self.config.client.batch_size
        epochs: int = config["local_epochs"]

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,
        )
        # Train the model using hyperparameters from config
        history = self.model.fit(
            x=X_train_l,
            y=X_train_l,
            batch_size=batch_size,
            epochs=start_epoch + epochs,
            validation_data=(X_val_l, X_val_l),
            callbacks=[tensorboard_callback],
            initial_epoch=start_epoch,
        )
        loss = history.history["total_loss"][0]
        val_loss = history.history["val_total_loss"][0]
        val_reconstruction_loss = history.history["val_rec_loss"][0]
        val_classification_loss = history.history["val_class_loss"][0]
        val_prox_loss = history.history["val_prox_loss"][0]

        self.last_val_loss = val_loss

        if self.config.setting == Setting.LOCAL and val_loss <= self.best_val_loss:
            print(
                f"[*] Round {start_epoch} best val loss so far: {val_loss}, saving model"
            )
            self.best_val_params = self.model.get_weights()
            self.best_val_loss = val_loss
            self.best_round = start_epoch

        # Calculate the threshold based on the local validation data
        ben_val = X_val[(y_val == 0)[:, 0]]
        rec = self.model.predict(ben_val)[:, :-1]
        mse = np.mean(np.power(ben_val - rec, 2), axis=1)
        self.threshold = get_threshold(ben_val, mse)

        cls_pred = self.model.predict(X_val)[:, -1]
        val_acc = float(((cls_pred > 0.5) == y_val).numpy().mean())

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        if self.config.setting == Setting.LOCAL:
            parameters_prime = [
                np.zeros_like(layer) for layer in self.model.get_weights()
            ]
        results = {
            "id": self.client_id,
            "loss": loss,
            "val_loss": val_loss,
            "val_classification_loss": val_classification_loss,
            "val_reconstruction_loss": val_reconstruction_loss,
            "val_prox_loss": val_prox_loss,
            "val_acc": val_acc,
            "scaler": self.scaler.dump(),
            "tracker": self.model.tracker.serialize(),
            "threshold": float(self.threshold),
            "num_mal_examples_train": int(num_examples_mal),
        }

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        if self.config.setting == Setting.FEDERATED:
            self.model.set_weights(parameters)
            self.threshold = config["threshold"]
            logger.info(f"Client {self.client_id} - threshold: {self.threshold}")
            if "scaler" in config:
                self.scaler = MinMaxScaler.load(config["scaler"])

        loss, eval_results = self.eval_test(self.X_test, self.y_test)
        eval_results["val_loss"] = self.last_val_loss

        num_examples = len(self.y_test)

        if self.config.setting == Setting.LOCAL:
            _, global_results = self.eval_test(
                self.global_X_test, self.global_y_test, "global_"
            )
            eval_results.update(global_results)
            if not self.config.fit_if_no_malware and not self.has_own_malware:
                # The training is done locally only and the client does not train on its own data
                # because it does not have any malware. Returning 0 as number of examples, so that
                # the results aggregation discards it
                num_examples = 0
            else:
                num_examples = len(self.global_y_test)

        return loss, num_examples, eval_results

    def eval_test(self, X_test, y_test, res_prefix=""):
        num_malicious = (y_test == 1).sum()

        X_test = self.scaler.transform(X_test)

        loss = 0.1

        y_pred_raw = self.model.predict(X_test)
        y_pred = (y_pred_raw[:, -1] > 0.5).numpy().T

        conf_matrix = confusion_matrix(y_test, y_pred)
        if conf_matrix.size == 1:
            conf_matrix = np.pad(conf_matrix, ((0, 1), (0, 1)), constant_values=0)
        cls_acc = float((y_pred == y_test).mean())

        num_malware = (y_test == 1).sum()
        num_benign = (y_test == 0).sum()

        cls_mal_mask = y_pred == 1
        cls_mal_true = y_test[cls_mal_mask]

        cls_fp = int((cls_mal_true == 0).sum())
        cls_tp = int((cls_mal_true == 1).sum())

        cls_ben_true = y_test[y_pred == 0]
        cls_tn = int((cls_ben_true == 0).sum())
        cls_fn = int((cls_ben_true == 1).sum())

        cls_tpr = (cls_tp / num_malware) if num_malicious else np.nan
        cls_fpr = cls_fp / num_benign

        rec = self.model.predict(X_test.astype("float32"))[:, :-1]
        mse = np.mean(np.power(X_test - rec, 2), axis=1)
        y_anomaly_pred = (mse > self.threshold).astype("float32")

        anomalies_mask = y_anomaly_pred == 1
        anomalies_true = y_test[anomalies_mask]

        ad_fp = int((anomalies_true == 0).sum())
        ad_tp = int((anomalies_true == 1).sum())
        ad_tn = int((anomalies_true == 0).sum())
        ad_fn = int((anomalies_true == 1).sum())

        ad_tpr = (ad_tp / num_malware) if num_malicious else np.nan
        ad_fpr = ad_fp / num_benign

        ad_accuracy = (y_anomaly_pred == y_test).mean()

        eval_results = {
            "id": self.client_id,
            "cls_fp": cls_fp,
            "cls_tp": cls_tp,
            "cls_tn": cls_tn,
            "cls_fn": cls_fn,
            "cls_tpr": cls_tpr,
            "cls_fpr": cls_fpr,
            "class_accuracy": cls_acc,
            "confusion_matrix": serialize(conf_matrix),
            "ad_fp": ad_fp,
            "ad_tp": ad_tp,
            "ad_tn": ad_tn,
            "ad_fn": ad_fn,
            "ad_tpr": ad_tpr,
            "ad_fpr": ad_fpr,
            "ad_acc": ad_accuracy,
        }

        return loss, {
            f"{res_prefix}{key}": value for key, value in eval_results.items()
        }

    def prepare_local_dataset(self, vaccine: np.array):
        X_train = self.scaler.transform(self.X_train)
        X_val = self.scaler.transform(self.X_val)
        if (
            self.y_train.sum() > 0 and not self.config.client.use_vaccine_if_own
        ) or vaccine.size == 0:
            return (
                X_train,
                X_val,
                self.y_train.reshape((-1, 1)),
                self.y_val.reshape((-1, 1)),
            )

        vaccine = self.scaler.transform(vaccine)
        mal_train, mal_val = train_test_split(
            vaccine,
            test_size=self.config.client.val_ratio,
            random_state=self.config.seed,
        )
        X_train, y_train = self._prepare_local_dataset(X_train, self.y_train, mal_train)
        X_val, y_val = self._prepare_local_dataset(X_val, self.y_val, mal_val)

        return X_train, X_val, y_train, y_val

    @staticmethod
    def _prepare_local_dataset(X, y, mal):
        shuffle_train = np.arange(X.shape[0] + mal.shape[0])
        np.random.shuffle(shuffle_train)

        # mal_ratio = y.mean()
        # if mal_ratio == 0.0:
        #     mal = np.resize(mal.to_numpy(), (int(X.shape[0] * 1.), X.shape[1]))
        X = np.concatenate([X, mal], axis=0)[shuffle_train]
        y = np.concatenate([y, np.ones(mal.shape[0])])[shuffle_train].reshape(-1, 1)
        return X, y

    @staticmethod
    def random_point_on_sphere(centroid, radius):
        rand = np.random.random(centroid.size) * 2 - 1
        norm = np.linalg.norm(rand)
        return centroid + (rand * radius / norm)


def load_partition(day: int, client_id: int, config: Config):
    """Load the training and test data to simulate a partition."""
    assert client_id in range(1, 11)
    assert day in range(1, 6)

    X_ben_train, X_ben_test = load_ben_data(
        day, client_id, config.data_dir, drop_labels=False
    )

    X_mal_train = load_day_dataset(
        config.client_malware(client_id, day), drop_labels=False
    )
    X_mal_test = load_day_dataset(
        config.client_malware(client_id, day + 1), drop_labels=False
    )

    X_train, X_val, y_train, y_val = create_supervised_dataset(
        X_ben_train, X_mal_train, 0.2, config.seed
    )
    X_test, _, y_test, _ = create_supervised_dataset(
        X_ben_test, X_mal_test, 0.0, config.seed
    )

    logger.info(f"[+] Num train samples for client {client_id}: {X_train.shape[0]}")
    logger.info(
        f"[+] Num of train malicious samples for client {client_id}: {(y_train == 1.0).sum()}"
    )
    logger.info(f"[+] Num of features for client {client_id}: {X_train.shape[1]}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def main(client_id: int, day: int, config_path: str = None, **overrides) -> None:
    config = Config.load(config_path, **overrides)
    data_dir = Path(config.data_dir)

    tf.keras.utils.set_random_seed(config.seed)

    if config.setting == Setting.LOCAL:
        _, _, global_X_test, _, _, global_y_test = load_centralized_data(day, config)
    else:
        global_X_test = global_y_test = None
    X_train, X_val, X_test, y_train, y_val, y_test = load_partition(
        day, client_id, config
    )

    disable_classifier = (y_train == 1).sum() < 1
    # Load and compile Keras model
    model = MultiHeadAutoEncoder(config)
    model.compile()

    if config.setting == Setting.LOCAL and day > 1 and config.load_model:
        model.load_weights(config.model_file(day - 1, client_id))

    if day > 1 and config.load and config.setting == Setting.LOCAL.value:
        logger.info("Loading local model from previous day")
        model.load_weights(config.local_model_file(day - 1, client_id))
    # Start Flower client
    client = ADClient(
        client_id,
        model,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        config,
        global_X_test,
        global_y_test,
    )

    with open(f"scaler_client_{client_id}.pckl", "wb") as f:
        pickle.dump(client.scaler, f)

    address = f"{config.ip_address}:{config.port}"
    fl.client.start_numpy_client(
        server_address=address,
        client=client,
        # root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )

    if config.setting == Setting.LOCAL.value:
        model.save_weights(config.local_model_file(day, client_id))

    if config.setting == Setting.LOCAL:
        model.set_weights(client.best_val_params)
        model.save_weights(config.model_file(day, client_id))


if __name__ == "__main__":
    Fire(main)
