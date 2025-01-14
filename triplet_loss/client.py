# SPDX-FileCopyrightText: 2021 Sebastian Garcia <sebastian.garcia@agents.fel.cvut.cz>
#  SPDX-License-Identifier: GPL-2.0-only
import warnings

warnings.filterwarnings("ignore")


import argparse
import os

import tensorflow as tf
import numpy as np
from pathlib import Path

import flwr as fl
from common.utils import get_threshold, serialize_array, deserialize_string, scale_data
from common.data_loading import load_mal_data, load_ben_data
from common.models import get_triplet_loss_model
from sklearn.model_selection import train_test_split

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class ADClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, X_test_ben, X_test_mal, X_pos, X_neg, seed):
        self.threshold = 100  # Or other high value
        self.model = model

        # Store the data unscaled
        self.X_train = X_train
        self.X_test_ben = X_test_ben
        self.X_test_mal = X_test_mal

        self.X_pos = X_pos
        self.X_neg = X_neg

        self.X_min = np.min(X_train, axis=0).values
        self.X_max = np.max(X_train, axis=0).values

        self.seed = seed

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
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        X_train = scale_data(self.X_train, self.X_min, self.X_max)
        X_train, X_val = train_test_split(
            X_train, test_size=0.2, random_state=self.seed
        )
        num_examples_train = len(X_train)

        X_pos = scale_data(self.X_pos, self.X_min, self.X_max)
        X_neg = scale_data(self.X_neg, self.X_min, self.X_max)

        # Train the model using hyperparameters from config
        history = self.model.fit(
            x=(X_train, X_pos[:num_examples_train], X_neg[:num_examples_train]),
            y=X_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(
                (X_val, X_pos[num_examples_train:], X_neg[num_examples_train:]),
                X_val,
            ),
        )

        loss = history.history["loss"][0]
        val_loss = history.history["val_loss"][0]

        if np.isnan(loss) or np.isnan(val_loss):
            print("[+] HERE!", np.isnan(loss), np.isnan(val_loss))
            # print("[+] HERE!", val_loss, self.max_loss)
            self.model.set_weights(parameters)

        # Calculate the threshold based on the local tarining data
        rec = self.model.predict((X_val, X_val, X_val))
        mse = np.mean(np.power(X_val - rec, 2), axis=1)
        self.threshold = get_threshold(X_val, mse)

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        results = {
            "loss": loss,
            # "accuracy": history.history["accuracy"][0],
            "val_loss": val_loss,
            # "val_accuracy": history.history["val_accuracy"][0],
            "threshold": float(self.threshold),
            "X_min": serialize_array(self.X_min),
            "X_max": serialize_array(self.X_max),
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        # steps: int = config["val_steps"]
        print("[+] Received global threshold:", config["threshold"])
        self.threshold = config["threshold"]

        self.X_min = np.array(deserialize_string(config["X_min"]))
        print("[+] Received global minimums")

        self.X_max = np.array(deserialize_string(config["X_max"]))
        print("[+] Received global maximums")

        X_test_ben_ = scale_data(self.X_test_ben, self.X_min, self.X_max)
        num_examples_test = len(X_test_ben_)

        X_pos = scale_data(self.X_pos, self.X_min, self.X_max)
        X_neg = scale_data(self.X_neg, self.X_min, self.X_max)

        X_pos = np.resize(X_pos, (num_examples_test, 36))
        X_neg = np.resize(X_neg, (num_examples_test, 36))

        # Evaluate global model parameters on the local test data and return results
        loss = self.model.evaluate(
            (X_test_ben_, X_pos, X_neg), X_test_ben_, 64, steps=None
        )

        rec_ben = self.model.predict((X_test_ben_, X_test_ben_, X_test_ben_))
        mse_ben = np.mean(np.power(X_test_ben_ - rec_ben, 2), axis=1)

        rec_mal = dict()
        mse_mal = dict()
        for folder in list(self.X_test_mal.keys()):
            X_test_mal_ = scale_data(self.X_test_mal[folder], self.X_min, self.X_max)
            rec_mal[folder] = self.model.predict(
                (X_test_mal_, X_test_mal_, X_test_mal_)
            )
            mse_mal[folder] = np.mean(
                np.power(X_test_mal_ - rec_mal[folder], 2), axis=1
            )

        # Detect all the samples which are anomalies.
        anomalies_ben = sum(mse_ben > self.threshold)
        anomalies_mal = 0
        for folder in list(self.X_test_mal.keys()):
            anomalies_mal += sum(mse_mal[folder] > self.threshold)

        num_malware = 0
        for folder in list(self.X_test_mal.keys()):
            num_malware += self.X_test_mal[folder].shape[0]

        fp = int(anomalies_ben)
        tp = int(anomalies_mal)
        tn = num_examples_test - fp
        # fn = num_malware - tp

        accuracy = (tp + tn) / (num_examples_test + num_malware)
        tpr = tp / num_malware
        fpr = fp / num_examples_test

        return (
            loss,
            int(num_examples_test + num_malware),
            {
                "anomalies_ben": int(anomalies_ben),
                "anomalies_mal": int(anomalies_mal),
                "accuracy": accuracy,
                "tpr": tpr,
                "fpr": fpr,
            },
        )


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--client_id", type=int, choices=range(1, 11), required=True)
    parser.add_argument("--day", type=int, choices=(range(1, 6)), required=True)
    parser.add_argument("--seed", type=int, required=False, default=8181)
    parser.add_argument("--ip_address", type=str, required=False, default="localhost")
    parser.add_argument("--port", type=int, required=False, default="8080")
    parser.add_argument("--data_dir", type=str, required=False, default="/data")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    # Load and compile Keras model
    model = get_triplet_loss_model()

    X_train, X_test_ben, X_test_mal, X_pos, X_neg = load_partition(
        args.day, args.client_id, Path(args.data_dir)
    )

    # Start Flower client
    client = ADClient(model, X_train, X_test_ben, X_test_mal, X_pos, X_neg, args.seed)

    fl.client.start_numpy_client(
        server_address=f"{args.ip_address}:{args.port}",
        client=client,
        # root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


def load_partition(day: int, client_id: int, data_dir: Path):
    """Load the training and test data to simulate a partition."""
    assert client_id in range(1, 11)
    assert day in range(1, 6)

    X_train, X_test_ben = load_ben_data(day, client_id, data_dir)
    neg_df = load_mal_data(day, data_dir)["CTU-Malware-Capture-Botnet-67-1"]
    X_test_mal = load_mal_data(1, data_dir)

    num_samples = len(X_train)
    num_neg_samples = len(neg_df)
    # repeats = num_samples // 15

    idx = np.random.randint(0, num_samples, 10)
    pos_samples = X_train.values[idx]
    pos_samples = np.resize(pos_samples, (num_samples, 36))
    np.random.shuffle(pos_samples)

    neg_samples = np.resize(neg_df.values, (num_samples, 36))
    np.random.shuffle(neg_samples)

    print(f"[+] Num train samples for client {client_id}: {X_train.shape[0]}")
    print(f"[+] Num of negative samples for client {client_id}: {num_neg_samples}")
    print(f"[+] Num of features for client {client_id}: {X_train.shape[1]}")
    print(f"[+] Num of positive samples for client {client_id}: {pos_samples.shape[0]}")
    print(f"[+] Num of negative for client {client_id}: {neg_samples.shape[0]}")

    return X_train, X_test_ben, X_test_mal, pos_samples, neg_samples


if __name__ == "__main__":
    main()
