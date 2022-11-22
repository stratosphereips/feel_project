import datetime
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

from common.config import Config

warnings.filterwarnings('ignore')

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
from pathlib import Path

import flwr as fl
from flwr.common.logger import logger
from common.data_loading import load_mal_data, load_ben_data, create_supervised_dataset, load_day_dataset
from common.models import get_classification_model, MultiHeadAutoEncoder
from common.utils import get_threshold, serialize_array, \
    deserialize_string, client_malware_map, MinMaxScaler, serialize, deserialize
from fire import Fire
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import logging

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class ADClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, X_train, X_val, X_test, y_train, y_val, y_test, config):
        self.client_id = client_id
        self.log = logging.getLogger(f'Client {client_id}')
        self.model = model
        self.log_dir = Path(f"../logs/fit/client{self.client_id:02}") / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Store the data unscaled
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.scaler = MinMaxScaler()
        self.scaler.fit(self.X_train)

        self.seed = config.seed
        self.proxy_radius_mult = config.client.proxy_radius_mult
        self.config = config

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        mal_dataset = deserialize(config['mal_dataset'])

        X_train, X_val, y_train, y_val = self.prepare_local_dataset(mal_dataset)
        X_train_l = np.concatenate([X_train, y_train.reshape((-1, 1))], axis=1).astype('float32')
        X_val_l = np.concatenate([X_val, y_val.reshape((-1, 1))], axis=1).astype('float32')

        proxy_spheres = deserialize(config['proxy_spheres'])
        if proxy_spheres:
            self.model.set_spheres(proxy_spheres)

        num_examples_train = len(X_train_l)
        num_examples_mal = y_train.sum()

        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        start_epoch: int = config['start_epoch']
        batch_size: int = self.config.client.batch_size
        epochs: int = config["local_epochs"]
        # neg_data = deserialize(config['neg_dataset'])
        # print(f"Got Neg data of size {neg_data.size}")

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir, histogram_freq=1,
        )
        # Train the model using hyperparameters from config
        history = self.model.fit(
            x=X_train_l,
            y=X_train_l,
            batch_size=batch_size,
            epochs=start_epoch + epochs,
            validation_data=(X_val_l, X_val_l),
            callbacks=[tensorboard_callback],
            initial_epoch=start_epoch
        )

        embedded = self.model.embed(X_train_l[:, :-1]).numpy()
        true_spheres, proxy_spheres = self.get_class_spheres(embedded, y_train[:, 0])
        self.model.set_local_spheres(true_spheres)
        #print(f"New spheres: {proxy_spheres}")

        loss = history.history["total_loss"][0]
        val_loss = history.history["val_total_loss"][0]
        val_reconstruction_loss = history.history['val_rec_loss'][0]
        val_classification_loss = history.history['val_class_loss'][0]
        val_positive_loss = history.history['val_+loss'][0]
        val_negative_loss = history.history['val_+loss'][0]
        val_prox_loss = history.history['val_prox_loss'][0]


        # Calculate the threshold based on the local validation data
        ben_val = X_val[(y_val == 0)[:, 0]]
        rec = self.model.predict(ben_val)[:, :-1]
        mse = np.mean(np.power(ben_val - rec, 2), axis=1)
        self.threshold = get_threshold(ben_val, mse)

        cls_pred = self.model.predict(X_val)[:, -1]
        val_acc = ((cls_pred > 0.5) == y_val).mean()

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        results = {
            "id": self.client_id,
            "loss": loss,
            "val_loss": val_loss,
            'val_classification_loss': val_classification_loss,
            'val_reconstruction_loss': val_reconstruction_loss,
            'val_positive_loss': val_positive_loss,
            'val_negative_loss': val_negative_loss,
            'val_prox_loss': val_prox_loss,
            'val_acc': val_acc,
            "scaler": self.scaler.dump(),
            "proxy_spheres": serialize(proxy_spheres),
            "tracker": self.model.tracker.serialize(),
            "threshold": float(self.threshold),
            "num_mal_examples_train": int(num_examples_mal)
        }

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        num_malicious = (self.y_test == 1).sum()

        # Update local model with global parameters
        self.model.set_weights(parameters)
        #log(f'Client {self.client_id} - classifier disabled: {self.model.disable_classifier}')

        # Get config values
        steps: int = config["val_steps"]
        self.threshold = config['threshold']
        logger.info(f'Client {self.client_id} - threshold: {self.threshold}')
        if 'scaler' in config:
            self.scaler = MinMaxScaler.load(config["scaler"])

        X_test = self.scaler.transform(self.X_test)
        num_examples_test = len(X_test)

        Xy = np.concatenate([X_test, self.y_test.reshape(-1, 1)], axis=1).astype('float32')
        loss, _, _, _, _, _, _ = self.model.evaluate(Xy, Xy)

        y_pred_raw = self.model.predict(X_test)
        y_pred = (y_pred_raw[:, -1] > 0.5).astype(float).T

        conf_matrix = confusion_matrix(self.y_test, y_pred)
        if conf_matrix.size == 1:
            conf_matrix = np.pad(conf_matrix, ((0, 1), (0, 1)), constant_values=0)
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

        rec = self.model.predict(X_test.astype('float32'))[:, :-1]
        mse = np.mean(np.power(X_test - rec, 2), axis=1)
        y_anomaly_pred = (mse > self.threshold).astype('float32')

        anomalies_mask = y_anomaly_pred == 1
        anomalies_true = self.y_test[anomalies_mask]

        ad_fp = int((anomalies_true == 0).sum())
        ad_tp = int((anomalies_true == 1).sum())

        ad_tpr = (ad_tp / num_malware) if num_malicious else np.nan
        ad_fpr = ad_fp / num_benign

        ad_accuracy = (y_anomaly_pred == self.y_test).mean()
        eval_results = {
            "cls_fp": cls_fp,
            "cls_tp": cls_tp,
            'cls_tn': cls_tn,
            'cls_fn': cls_fn,
            'cls_tpr': cls_tpr,
            'cls_fpr': cls_fpr,
            "class_accuracy": cls_acc,
            "confusion_matrix": serialize(conf_matrix),
            'ad_fp': ad_fp,
            'ad_tp': ad_tp,
            'ad_tpr': ad_tpr,
            'ad_fpr': ad_fpr,
            'ad_acc': ad_accuracy,
        }

        return loss, int(num_examples_test), eval_results

    def prepare_local_dataset(self, vaccine: np.array):
        X_train = self.scaler.transform(self.X_train)
        X_val = self.scaler.transform(self.X_val)
        vaccine = self.scaler.transform(vaccine)
        if self.y_train.sum() > 0 and not self.config.client.use_vaccine_if_own:
            return X_train, X_val, self.y_train.reshape((-1, 1)), self.y_val.reshape((-1, 1))

        mal_train, mal_val = train_test_split(
            vaccine,
            test_size=self.config.client.val_ratio,
            random_state=self.config.seed
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

    def get_class_spheres(self, embedded, y):
        true_spheres = {}
        proxy_spheres = {}
        for cls in np.unique(y).tolist():
            embedded_cls = embedded[y == cls]
            centroid = embedded_cls.mean(axis=0)
            radius = np.linalg.norm(embedded_cls - centroid, axis=1).max()
            true_spheres[cls] = (centroid, radius)
            #print(f"Client {self.client_id}, {cls=}, {radius=}")
            proxy_radius = radius + max(radius * self.proxy_radius_mult, 0.1)
            proxy_center = self.random_point_on_sphere(centroid, proxy_radius)
            proxy_spheres[cls] = (proxy_center, proxy_radius)
        return true_spheres, proxy_spheres

    @staticmethod
    def random_point_on_sphere(centroid, radius):
        rand = np.random.random(centroid.size) * 2 - 1
        norm = np.linalg.norm(rand)
        return centroid + (rand * radius / norm)


def load_partition(day: int, client_id: int, config: Config):
    """Load the training and test data to simulate a partition."""
    assert client_id in range(1, 11)
    assert day in range(1, 6)

    X_ben_train, X_ben_test = load_ben_data(day, client_id, config.data_dir, drop_labels=False)

    X_mal_train = load_day_dataset(config.client_malware(client_id, day), drop_labels=False)
    X_mal_test = load_day_dataset(config.client_malware(client_id, day+1), drop_labels=False)

    X_train, X_val, y_train, y_val = create_supervised_dataset(X_ben_train, X_mal_train, 0.2, config.seed)
    X_test, _, y_test, _ = create_supervised_dataset(X_ben_test, X_mal_test, 0.0, config.seed)

    logger.info(f"[+] Num train samples for client {client_id}: {X_train.shape[0]}")
    logger.info(f"[+] Num of train malicious samples for client {client_id}: {(y_train == 1.0).sum()}")
    logger.info(f"[+] Num of features for client {client_id}: {X_train.shape[1]}")

    return X_train, X_val, X_test, y_train, y_val, y_test

# def load_partition(day: int, client_id: int, data_dir: Path, seed):
#     """Load the training and test data to simulate a partition."""
#     assert client_id in range(1, 11)
#     assert day in range(1, 6)
#
#     X_ben_train, X_ben_test = load_ben_data(day, client_id, data_dir, drop_labels=False)
#     if day == 1:
#         client_malwares = {client_malware_map[client_id], client_malware_map[5]}
#     else:
#         client_malwares = {client_malware_map[client_id]}
#     print(f"====\n\n\n Reading MAL DATA from {client_malwares}\n\n\n")
#
#     mal_tr = load_mal_data(day, data_dir, drop_labels=False)
#     mal_ts = load_mal_data(day+1, data_dir, drop_labels=False)
#     X_mal_train = pd.concat([mal_tr[mal] for mal in client_malwares]) #if client_id in client_malware_map else pd.DataFrame()
#     print(X_mal_train.shape)
#     X_mal_test = pd.concat([mal_ts[mal] for mal in client_malwares]) #if client_id in client_malware_map else pd.DataFrame() #[client_malware_map[client_id]]
#
#     X_train, X_val, y_train, y_val = create_supervised_dataset(X_ben_train, X_mal_train, 0.2, seed)
#     X_test, _, y_test, _ = create_supervised_dataset(X_ben_test, X_mal_test, 0.0, seed)
#
#     print(f"[+] Num train samples for client {client_id}: {X_train.shape[0]}")
#     print(f"[+] Num of train malicious samples for client {client_id}: {(y_train == 1.0).sum()}")
#     print(f"[+] Num of features for client {client_id}: {X_train.shape[1]}")
#
#     return X_train, X_val, X_test, y_train, y_val, y_test


def main(client_id: int, day: int, config_path: str = None, **overrides) -> None:
    config = Config.load(config_path, **overrides)
    # Parse command line arguments
    data_dir = Path(config.data_dir)

    tf.keras.utils.set_random_seed(config.seed)

    # X_train, X_val, X_test, y_train, y_val, y_test = load_partition(day, client_id, config.data_dir, config.seed)
    X_train, X_val, X_test, y_train, y_val, y_test = load_partition(day, client_id, config)


    disable_classifier = (y_train == 1).sum() < 1
    # Load and compile Keras model
    model = MultiHeadAutoEncoder(config)
    model.set_spheres({-1: {0.0: (np.zeros(10), 1.0), 1.0: (np.ones(10), 1.0)}})
    model.set_local_spheres({0.0: (np.zeros(10), 1.0), 1.0: (np.ones(10), 1.0)})
    model.compile()

    # Start Flower client
    client = ADClient(client_id, model, X_train, X_val, X_test, y_train, y_val, y_test, config)

    with open(f'scaler_client_{client_id}.pckl', 'wb') as f:
        pickle.dump(client.scaler, f)

    address = f"{config.ip_address}:{config.port}"
    fl.client.start_numpy_client(
        server_address=address,
        client=client,
        # root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


if __name__ == "__main__":
    Fire(main)
