import warnings
warnings.filterwarnings('ignore')


import argparse
import os
from pathlib import Path

import tensorflow as tf
import numpy as np

import flwr as fl
from utils import get_ben_data, get_mal_data, get_model, get_threshold
from sklearn import preprocessing
# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class ADClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, X_test_ben, X_test_mal):
        self.threshold = 100 # Or other high value
        self.model = model
        # (self.X_train, self.X_min, self.X_max) = self.scale_data(X_train)
        # self.X_test_ben = self.scale_transform(X_test_ben)
        
        scaler = preprocessing.MinMaxScaler().fit(X_train)
        self.X_train = scaler.transform(X_train)
        self.X_test_ben = scaler.transform(X_test_ben)

        self.X_test_mal = dict()
        for folder in list(X_test_mal.keys()):
            self.X_test_mal[folder] = scaler.transform(X_test_mal[folder])
       
        # for folder in list(X_test_mal.keys()):
            # self.X_test_mal[folder] = self.scale_transform(X_test_mal[folder])

    def scale_data(self, X):
        # Min max scaling"
        X_min = np.min(X, axis=0).values
        X_max = np.max(X, axis=0).values
        # Numerical stability requires some small value in the denom.
        X_std = (X - X_min) / (X_max - X_min + 0.000001)
        return (X_std, X_min, X_max)

    def scale_transform(self, X):
        # Transform the test data
        X_std = (X - self.X_min) / (self.X_max - self.X_min + 0.000001)
        return X_std

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

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.X_train,
            self.X_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Calculate the threshold based on the local tarining data
        rec = self.model.predict(self.X_train)
        mse = np.mean(np.power(self.X_train - rec, 2), axis=1)
        self.threshold = get_threshold(self.X_train, mse)

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.X_train)
        results = {
            "loss": history.history["loss"][0],
            # "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            # "val_accuracy": history.history["val_accuracy"][0],
            "threshold": float(self.threshold)
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]
        print("[+] Received global threshold:", config["threshold"])
        self.threshold = config["threshold"]

        # Evaluate global model parameters on the local test data and return results
        loss = self.model.evaluate(self.X_test_ben, self.X_test_ben, 32, steps=None)
        num_examples_test = len(self.X_test_ben)

        rec_ben = self.model.predict(self.X_test_ben)
        mse_ben = np.mean(np.power(self.X_test_ben - rec_ben, 2), axis=1)

        rec_mal = dict()
        mse_mal = dict()
        for folder in list(self.X_test_mal.keys()):
            rec_mal[folder] = self.model.predict(self.X_test_mal[folder])
            mse_mal[folder] = np.mean(np.power(self.X_test_mal[folder] - rec_mal[folder], 2), axis=1)

        # Detect all the samples which are anomalies.
        anomalies_ben = sum(mse_ben > self.threshold)
        anomalies_mal = 0
        for folder in list(self.X_test_mal.keys()):
            anomalies_mal += sum(mse_mal[folder] > self.threshold)

        return loss, num_examples_test, {"anomalies_ben": int(anomalies_ben), 
                                        "anomalies_mal": int(anomalies_mal)}


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--client_id", type=int, choices=range(1, 11), required=True)
    parser.add_argument("--day", type=int, choices=(range(1,6)), required=True)
    args = parser.parse_args()

    # Load and compile Keras model
    model = get_model()

    X_train, X_test_ben, X_test_mal = load_partition(args. day, args.client_id)

    # Start Flower client
    client = ADClient(model, X_train, X_test_ben, X_test_mal)

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client,
        root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


def load_partition(day: int, client_id: int):
    """Load 1/5th of the training and test data to simulate a partition."""
    assert client_id in range(1, 11)
    assert day in range(1, 6)

    X_train, X_test_ben = get_ben_data(day, client_id)
    X_test_mal = get_mal_data()

    # num_samples = x_train.shape[0] // 5
    print(f"[+] Num train samples for client{client_id}: {X_train.shape[0]}")
    return X_train, X_test_ben, X_test_mal
    # return x_train[idx * num_samples : (idx + 1) * num_samples], x_test[idx * num_samples : (idx + 1) * num_samples]


if __name__ == "__main__":
    main()
