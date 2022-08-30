from gc import callbacks
import warnings
warnings.filterwarnings('ignore')


import argparse
import os
from pathlib import Path

import tensorflow as tf
import numpy as np

import flwr as fl
from utils import get_ben_data, get_mal_data, get_model, get_threshold,serialize_array, deserialize_string, scale_data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class ADClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, X_test_ben, X_test_mal, seed, checkpoint_filepath):
        self.threshold = 100 # Or other high value
        self.model = model
        
        # Store the data unscaled
        self.X_train = X_train
        self.X_test_ben = X_test_ben
        self.X_test_mal = X_test_mal

        self.X_min = np.min(X_train, axis=0).values
        self.X_max = np.max(X_train, axis=0).values

        self.seed=seed

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
        X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=self.seed)

        # Train the model using hyperparameters from config
        history = self.model.fit(
            X_train,
            X_train,
            batch_size,
            epochs,
            validation_data=(X_val, X_val)
        )

        loss = history.history["loss"][0]
        val_loss = history.history["val_loss"][0]

        if np.isnan(loss) or np.isnan(val_loss):
            print("[+] HERE!", np.isnan(loss), np.isnan(val_loss))
            print("[+] HERE!", val_loss, self.max_loss)
            self.model.set_weights(parameters)
        
        # Calculate the threshold based on the local tarining data
        rec = self.model.predict(X_val)
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
            "X_min": serialize_array(self.X_min),
            "X_max": serialize_array(self.X_max)
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

        self.X_min = np.array(deserialize_string(config["X_min"]))
        print("[+] Received global minimums")

        self.X_max = np.array(deserialize_string(config["X_max"]))
        print("[+] Received global maximums")

        X_test_ben_ = scale_data(self.X_test_ben, self.X_min, self.X_max)

        # Evaluate global model parameters on the local test data and return results
        loss = self.model.evaluate(X_test_ben_, X_test_ben_, 64, steps=None)
        num_examples_test = len(X_test_ben_)

        rec_ben = self.model.predict(X_test_ben_)
        mse_ben = np.mean(np.power(X_test_ben_ - rec_ben, 2), axis=1)

        rec_mal = dict()
        mse_mal = dict()
        for folder in list(self.X_test_mal.keys()):
            X_test_mal_ = scale_data(self.X_test_mal[folder], self.X_min, self.X_max)
            rec_mal[folder] = self.model.predict(X_test_mal_)
            mse_mal[folder] = np.mean(np.power(X_test_mal_ - rec_mal[folder], 2), axis=1)

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
        fn = num_malware - tp

        accuracy = (tp + tn) / (num_examples_test + num_malware)
        tpr = tp / num_malware
        fpr = fp / num_examples_test


        return loss, int(num_examples_test+num_malware), {
                            "anomalies_ben": int(anomalies_ben), 
                            "anomalies_mal": int(anomalies_mal),
                            "accuracy": accuracy,
                            "tpr": tpr,
                            "fpr": fpr
                            }


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--client_id", type=int, choices=range(1, 11), required=True)
    parser.add_argument("--day", type=int, choices=(range(1,6)), required=True)
    parser.add_argument("--seed", type=int, required=False, default=8181)
    parser.add_argument("--ip_address", type=str, required=False, default="localhost")
    parser.add_argument("--port", type=int, required=False, default="8080")
    parser.add_argument("--data_dir", type=str, required=False, default="/data")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    # Load and compile Keras model
    model = get_model()

    X_train, X_test_ben, X_test_mal = load_partition(args.day, args.client_id, args.data_dir)

    # Start Flower client
    client = ADClient(model, X_train, X_test_ben, X_test_mal, args.seed, f'/tmp/checkpoint/client{args.client_id}')

    fl.client.start_numpy_client(
        server_address=f"{args.ip_address}:{args.port}",
        client=client,
        root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


def load_partition(day: int, client_id: int, data_dir: str):
    """Load the training and test data to simulate a partition."""
    assert client_id in range(1, 11)
    assert day in range(1, 6)

    X_train, X_test_ben = get_ben_data(day, client_id, data_dir)
    X_test_mal = get_mal_data(data_dir)

    print(f"[+] Num train samples for client{client_id}: {X_train.shape[0]}")
    print(f"[+] Num of features for client{client_id}: {X_train.shape[1]}")
    return X_train, X_test_ben, X_test_mal


if __name__ == "__main__":
    main()
