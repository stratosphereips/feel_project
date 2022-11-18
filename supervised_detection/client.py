import datetime
import warnings

import pandas as pd

warnings.filterwarnings('ignore')

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
from pathlib import Path

import flwr as fl
from common.data_loading import load_mal_data, load_ben_data, create_supervised_dataset
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
    def __init__(self, client_id, model, X_train, X_val, X_test, y_train, y_val, y_test, seed, proxy_diameter):
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

        self.seed = seed
        self.proxy_radius_mult = proxy_diameter

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        X_train = self.scaler.transform(self.X_train)
        X_val = self.scaler.transform(self.X_val)
        X_train = np.concatenate([X_train, self.y_train.reshape(-1, 1)], axis=1).astype('float32')
        X_val = np.concatenate([X_val, self.y_val.reshape(-1, 1)], axis=1).astype('float32')

        proxy_spheres = deserialize(config['proxy_spheres'])
        if proxy_spheres:
            self.model.set_spheres(proxy_spheres)

        num_examples_train = len(X_train)
        num_malicious = (self.y_train == 1.0).sum()

        # if num_malicious < 10:
        #     self.log.warning('Client has too few malicious examples for supervised training - skipping it.')
        #     return [], 0, {}
        # Update local model parameter
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        start_epoch: int = config['start_epoch']
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        neg_data = deserialize(config['neg_dataset'])
        print(f"Got Neg data of size {neg_data.size}")

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir, histogram_freq=1,
        )
        # Train the model using hyperparameters from config
        history = self.model.fit(
            x=X_train,
            y=X_train,
            batch_size=batch_size,
            epochs=start_epoch + epochs,
            validation_data=(X_val, X_val),
            callbacks=[tensorboard_callback],
            initial_epoch=start_epoch
        )

        embedded = self.model.embed(X_train[:, :-1]).numpy()
        true_spheres, proxy_spheres = self.get_class_spheres(embedded, self.y_train)
        self.model.set_local_spheres(true_spheres)
        print(f"New spheres: {proxy_spheres}")

        loss = history.history["total_loss"][0]
        val_loss = history.history["val_total_loss"][0]
        val_reconstruction_loss = history.history['val_reconstruction_loss'][0]
        val_classification_loss = history.history['val_classification_loss'][0]
        val_positive_loss = history.history['val_positive_loss'][0]
        val_negative_loss = history.history['val_negative_loss'][0]
        val_prox_loss = history.history['val_proximal_loss'][0]
        if np.isnan(loss) or np.isnan(val_loss):
            print("[+] HERE!", np.isnan(loss), np.isnan(val_loss))
            # print("[+] HERE!", val_loss, self.max_loss)
            self.model.set_weights(parameters)

        # X_val_pred =
        # self.threshold = get_threshold(X_val, mse)

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        results = {
            "id": self.client_id,
            "loss": loss,
            # "accuracy": history.history["accuracy"][0],
            "val_loss": val_loss,
            'val_classification_loss': val_classification_loss,
            'val_reconstruction_loss': val_reconstruction_loss,
            'val_positive_loss': val_positive_loss,
            'val_negative_loss': val_negative_loss,
            'val_prox_loss': val_prox_loss,
            # "val_accuracy": history.history["val_accuracy"][0],
            "scaler": self.scaler.dump(),
            "proxy_spheres": serialize(proxy_spheres),
            "tracker": self.model.tracker.serialize()
        }

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        num_malicious = (self.y_test == 1).sum()

        # if num_malicious == 0:
        #     self.log.warning('Client has too few malicious examples for evaluation - skipping it.')
        #     return 0.0, 0, {}
        # Update local model with global parameters
        self.model.set_weights(parameters)
        print(f'Client {self.client_id} - classifier disabled: {self.model.disable_classifier}')

        # Get config values
        steps: int = config["val_steps"]
        self.threshold = config['threshold']
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

        cls_tpr = (cls_tp / num_malware) if num_malicious else np.nan
        cls_fpr = cls_fp / num_benign

        X_val = self.scaler.transform(self.X_val)
        X_val_ben = X_val[self.y_val == 0]
        rec_val = self.model.predict(X_val_ben)[:, :-1]
        mse_val = np.mean(np.power(X_val_ben - rec_val, 2), axis=1)

        th = get_threshold(X_val_ben, mse_val)

        rec = self.model.predict(X_test.astype('float32'))[:, :-1]
        mse = np.mean(np.power(X_test - rec, 2), axis=1)
        y_anomaly_pred = (mse > th).astype('float32')

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
            'cls_tpr': cls_tpr,
            'cls_fpr': cls_fpr,
            "class_accuracy": cls_acc,
            "confusion_matrix": serialize(conf_matrix),
            'ad_th': th,
            'ad_fp': ad_fp,
            'ad_tp': ad_tp,
            'ad_tpr': ad_tpr,
            'ad_fpr': ad_fpr,
            'ad_acc': ad_accuracy,
        }

        return loss, int(num_examples_test), eval_results

    def get_class_spheres(self, embedded, y):
        true_spheres = {}
        proxy_spheres = {}
        for cls in set(y):
            embedded_cls = embedded[y == cls]
            centroid = embedded_cls.mean(axis=0)
            radius = np.linalg.norm(embedded_cls - centroid, axis=1).max()
            true_spheres[cls] = (centroid, radius)
            print(f"Client {self.client_id}, {cls=}, {radius=}")
            proxy_radius = radius + max(radius * self.proxy_radius_mult, 0.1)
            proxy_center = self.random_point_on_sphere(centroid, proxy_radius)
            proxy_spheres[cls] = (proxy_center, proxy_radius)
        return true_spheres, proxy_spheres

    @staticmethod
    def random_point_on_sphere(centroid, radius):
        rand = np.random.random(centroid.size) * 2 - 1
        norm = np.linalg.norm(rand)
        return centroid + (rand * radius / norm)


def load_partition(day: int, client_id: int, data_dir: Path, seed):
    """Load the training and test data to simulate a partition."""
    assert client_id in range(1, 11)
    assert day in range(1, 6)

    X_ben_train, X_ben_test = load_ben_data(day, client_id, data_dir, drop_labels=False)
    if day == 1:
        client_malwares = {client_malware_map[client_id], client_malware_map[5]}
    else:
        client_malwares = {client_malware_map[client_id]}

    mal_tr = load_mal_data(day, data_dir, drop_labels=False)
    mal_ts = load_mal_data(day+1, data_dir, drop_labels=False)
    X_mal_train = pd.concat([mal_tr[mal] for mal in client_malwares]) #if client_id in client_malware_map else pd.DataFrame()
    print(X_mal_train.shape)
    X_mal_test = pd.concat([mal_ts[mal] for mal in client_malwares]) #if client_id in client_malware_map else pd.DataFrame() #[client_malware_map[client_id]]

    X_train, X_val, y_train, y_val = create_supervised_dataset(X_ben_train, X_mal_train, 0.2, seed)
    X_test, _, y_test, _ = create_supervised_dataset(X_ben_test, X_mal_test, 0.0, seed)

    print(f"[+] Num train samples for client {client_id}: {X_train.shape[0]}")
    print(f"[+] Num of train malicious samples for client {client_id}: {(y_train == 1.0).sum()}")
    print(f"[+] Num of features for client {client_id}: {X_train.shape[1]}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def main(client_id: int, day: int, model_path: str, seed: int = 8181, ip_address: str = '0.0.0.0', port: int = 8000,
         data_dir: str = '../data') -> None:
    # Parse command line arguments
    data_dir = Path(data_dir)
    model_path = Path(model_path)

    tf.keras.utils.set_random_seed(seed)

    X_train, X_val, X_test, y_train, y_val, y_test = load_partition(day, client_id, Path(data_dir), seed)

    disable_classifier = (y_train == 1).sum() < 1
    # Load and compile Keras model
    model = MultiHeadAutoEncoder(disable_classifier=disable_classifier, proximal=True)
    model.set_spheres({-1: {0.0: (np.zeros(10), 1.0), 1.0: (np.ones(10), 1.0)}})
    model.set_local_spheres({0.0: (np.zeros(10), 1.0), 1.0: (np.ones(10), 1.0)})
    model.compile()

    # Start Flower client
    client = ADClient(client_id, model, X_train, X_val, X_test, y_train, y_val, y_test, seed, proxy_diameter=1.1)

    with open(f'scaler_client_{client_id}.pckl', 'wb') as f:
        pickle.dump(client.scaler, f)

    address = f"{ip_address}:{port}"
    fl.client.start_numpy_client(
        server_address=address,
        client=client,
        # root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


if __name__ == "__main__":
    Fire(main)
