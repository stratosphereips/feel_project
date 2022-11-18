from gc import callbacks
import warnings

import fire
from sklearn.metrics import confusion_matrix

from common.config import Config

warnings.filterwarnings('ignore')

import argparse
from pathlib import Path

import tensorflow as tf
import numpy as np

import flwr as fl
from common.utils import get_threshold, serialize_array, deserialize_string, scale_data, MinMaxScaler, serialize, client_malware_map
from common.data_loading import load_mal_data, load_ben_data
from common.models import get_ad_model, MultiHeadAutoEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class ADClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, X_test_ben, X_test_mal, client_id, config: Config):
        self.threshold = 100  # Or other high value
        self.model = model

        # Store the data unscaled
        self.X_train = X_train
        self.X_test_ben = X_test_ben
        self.X_test_mal = X_test_mal

        self.scaler = MinMaxScaler()
        self.scaler.fit(self.X_train)

        self.config = config
        self.seed = config.seed
        self.client_id = client_id

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        print(config)
        batch_size: int = config["batch_size"]
        start_epoch: int = config['start_epoch']
        epochs: int = config["local_epochs"]

        X_train = self.scaler.transform(self.X_train)
        X_train, X_val = train_test_split(X_train, test_size=self.config.client.val_ratio, random_state=self.seed)
        X_train_l = np.hstack([X_train, np.zeros((X_train.shape[0], 1))])
        X_val_l = np.hstack([X_val, np.zeros((X_val.shape[0], 1))])

        # Train the model using hyperparameters from config
        history = self.model.fit(
            X_train_l,
            X_train_l,
            validation_data=(X_val_l, X_val_l),
            batch_size=batch_size,
            initial_epoch=start_epoch,
            epochs=start_epoch + epochs,
        )

        loss = history.history["total_loss"][0]
        val_loss = history.history["val_total_loss"][0]

        if np.isnan(loss) or np.isnan(val_loss):
            print("[+] HERE!", np.isnan(loss), np.isnan(val_loss))
            print("[+] HERE!", val_loss, self.max_loss)
            self.model.set_weights(parameters)

        # Calculate the threshold based on the local validation data
        X_val = X_val
        rec = self.model.predict(X_val)[:, :-1]
        mse = np.mean(np.power(X_val - rec, 2), axis=1)
        self.threshold = get_threshold(X_val, mse)

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(X_train)
        results = {
            "loss": loss,
            # "accuracy": history.history["accuracy"][0],
            "val_loss": val_loss,
            # "val_accuracy": history.history["val_accuracy"][0],
            "threshold": float(self.threshold),
            "id": self.client_id,
            "scaler": self.scaler.dump()
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        if 'scaler' in config:
            self.scaler = MinMaxScaler.load(config["scaler"])

        print("[+] Received global threshold:", config["threshold"])
        self.threshold = config["threshold"]

        X_test_ben_ = self.scaler.transform(self.X_test_ben)
        X_test_ben_l = np.hstack([X_test_ben_, np.zeros((X_test_ben_.shape[0], 1))])

        # Evaluate global model parameters on the local test data and return results
        loss, _, _, _, _, _, _ = self.model.evaluate(X_test_ben_l, X_test_ben_l, 64, steps=None)
        num_examples_test = len(X_test_ben_)

        rec_ben = self.model.predict(X_test_ben_)[:, :-1]
        mse_ben = np.mean(np.power(X_test_ben_ - rec_ben, 2), axis=1)

        rec_mal = dict()
        mse_mal = dict()

        for folder in list(self.X_test_mal.keys()):
            if folder != client_malware_map[self.client_id]:
                continue
            X_test_mal_ = self.scaler.transform(self.X_test_mal[folder])
            rec_mal[folder] = self.model.predict(X_test_mal_)[:, :-1]
            mse_mal[folder] = np.mean(np.power(X_test_mal_ - rec_mal[folder], 2), axis=1)

        # Detect all the samples which are anomalies.
        y_ben = mse_ben > self.threshold
        anomalies_ben = sum(y_ben)
        y_pred = y_ben.tolist()
        y_true = np.zeros_like(y_ben).tolist()
        anomalies_mal = 0
        for folder in list(self.X_test_mal.keys()):
            if folder != client_malware_map[self.client_id]:
                continue
            y_mal = mse_mal[folder] > self.threshold
            anomalies_mal += sum(y_mal)
            y_pred += y_mal.tolist()
            y_true += np.ones_like(y_mal).tolist()

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        num_malware = 0
        for folder in list(self.X_test_mal.keys()):
            num_malware += self.X_test_mal[folder].shape[0]

        fp = int(anomalies_ben)
        tp = int(anomalies_mal)
        tn = num_examples_test - fp
        fn = num_malware - tp

        accuracy = (tp + tn) / (num_examples_test + num_malware)
        tpr = tp / num_malware
        fpr = fp / num_examples_test

        conf_matrix = confusion_matrix(y_true, y_pred)

        return loss, int(num_examples_test + num_malware), {
            "anomalies_ben": int(anomalies_ben),
            "anomalies_mal": int(anomalies_mal),
            "accuracy": (y_true == y_pred).mean(),
            "tpr": tpr,
            "fpr": fpr,
            "confusion_matrix": serialize(conf_matrix)
        }


def main(day, client_id: int, config_path: str = None, **overrides) -> None:
    config = Config.load(config_path, **overrides)
    # Parse command line arguments
    tf.keras.utils.set_random_seed(config.seed)

    # Load and compile Keras model
    model = MultiHeadAutoEncoder(config)
    model.compile()

    X_train, X_test_ben, X_test_mal = load_partition(day, client_id, Path(config.data_dir))

    # Start Flower client
    client = ADClient(model, X_train, X_test_ben, X_test_mal, client_id, config)

    fl.client.start_numpy_client(
        server_address=f"{config.ip_address}:{config.port}",
        client=client,
        # root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


def load_partition(day: int, client_id: int, data_dir: Path):
    """Load the training and test data to simulate a partition."""
    assert client_id in range(1, 11)
    assert day in range(1, 6)

    X_train, X_test_ben = load_ben_data(day, client_id, data_dir)
    X_test_mal = load_mal_data(1, data_dir)

    print(f"[+] Num train samples for client{client_id}: {X_train.shape[0]}")
    print(f"[+] Num of features for client{client_id}: {X_train.shape[1]}")
    return X_train, X_test_ben, X_test_mal


if __name__ == "__main__":
    fire.Fire(main)
