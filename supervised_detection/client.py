import warnings

warnings.filterwarnings('ignore')

import os

import tensorflow as tf
import numpy as np
from pathlib import Path

import flwr as fl
from common.data_loading import load_mal_data, load_ben_data, create_supervised_dataset
from common.models import get_classification_model
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
    def __init__(self, client_id, model, X_train, X_val, X_test, y_train, y_val, y_test, seed):
        self.client_id = client_id
        self.log = logging.getLogger(f'Client {client_id}')
        self.model = model

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

        num_examples_train = len(X_train)
        num_malicious = (self.y_train == 1.0).sum()

        if num_malicious < 10:
            self.log.warning('Client has too few malicious examples for supervised training - skipping it.')
            return [], 0, {}
        # Update local model parameter
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        neg_data = deserialize(config['neg_dataset'])
        print(f"Got Neg data of size {neg_data.size}")


        # Train the model using hyperparameters from config
        history = self.model.fit(
            x=X_train,
            y=self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, self.y_val)
        )

        loss = history.history["loss"][0]
        val_loss = history.history["val_loss"][0]

        if np.isnan(loss) or np.isnan(val_loss):
            print("[+] HERE!", np.isnan(loss), np.isnan(val_loss))
            # print("[+] HERE!", val_loss, self.max_loss)
            self.model.set_weights(parameters)

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()

        results = {
            "loss": loss,
            #"accuracy": history.history["accuracy"][0],
            "val_loss": val_loss,
            #"val_accuracy": history.history["val_accuracy"][0],
            "scaler": self.scaler.dump()
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        num_malicious = (self.y_test == 1).sum()

        if num_malicious == 0:
            self.log.warning('Client has too few malicious examples for evaluation - skipping it.')
            return 0.0, 0, {}
        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        # steps: int = config["val_steps"]
        if 'scaler' in config:
            self.scaler = MinMaxScaler.load(config["scaler"])

        X_test = self.scaler.transform(self.X_test)
        num_examples_test = len(X_test)

        # Evaluate global model parameters on the local test data and return results
        loss = self.model.evaluate(X_test, self.y_test, 64, steps=None)

        y_pred = self.model.predict(X_test)
        print(y_pred)
        y_pred = (y_pred > 0.5).astype(float).T[0]

        report = classification_report(self.y_test, y_pred, target_names=['Benign', 'Malicious'], output_dict=True)
        conf_matrix = confusion_matrix(self.y_test, y_pred)

        num_malware = (self.y_test == 1).sum()

        anomalies_mask = y_pred == 1
        anomalies_true = self.y_test[anomalies_mask]

        fp = int((anomalies_true == 0).sum())
        tp = int((anomalies_true == 1).sum())
        tn = num_examples_test - fp

        tpr = tp / num_malware
        fpr = fp / num_malware

        eval_results = {
            "anomalies_ben": fp,
            "anomalies_mal": tp,
            'tpr': tpr,
            'fpr': fpr,
            "accuracy": report['accuracy'],
            "num_benign": report['Benign']['support'],
            "num_malicious": report['Malicious']['support'],
            "precision": report["weighted avg"]['precision'],
            "recall": report["weighted avg"]['recall'],
            "f1": report["weighted avg"]['f1-score'],
            "sensitivity": report['Benign']['recall'],
            "specificity": report['Malicious']['recall'],
            "confusion_matrix": serialize(conf_matrix)
        }

        return loss, int(num_examples_test + num_malware), eval_results


def main(client_id: int, day: int, model_path: str, seed: int = 8181, ip_address: str = '0.0.0.0', port: int = 8000,
         data_dir: str = '../data') -> None:
    # Parse command line arguments
    data_dir = Path(data_dir)
    model_path = Path(model_path)

    tf.keras.utils.set_random_seed(seed)

    # Load and compile Keras model
    model = get_classification_model(model_path, encoder_lr=1e-4, classifier_lr=5e-3, dropout=0.0)


    X_train, X_val, X_test, y_train, y_val, y_test = load_partition(day, client_id, Path(data_dir), seed)

    # Start Flower client
    client = ADClient(client_id, model, X_train, X_val, X_test, y_train, y_val, y_test, seed)

    with open(f'scaler_client_{client_id}.pckl', 'wb') as f:
        pickle.dump(client.scaler, f)

    address = f"{ip_address}:{port}"
    fl.client.start_numpy_client(
        server_address=address,
        client=client,
        # root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


def load_partition(day: int, client_id: int, data_dir: Path, seed):
    """Load the training and test data to simulate a partition."""
    assert client_id in range(1, 11)
    assert day in range(1, 6)

    X_ben_train, X_ben_test = load_ben_data(day, client_id, data_dir, drop_labels=False)
    X_mal_train = load_mal_data(day, data_dir, drop_labels=False)[client_malware_map[client_id]]
    X_mal_test = load_mal_data(day+1, data_dir, drop_labels=False)[client_malware_map[client_id]]

    X_train, X_val, y_train, y_val = create_supervised_dataset(X_ben_train, X_mal_train, 0.2, seed)
    X_test, _, y_test, _ = create_supervised_dataset(X_ben_test, X_mal_test, 0.0, seed)


    print(f"[+] Num train samples for client {client_id}: {X_train.shape[0]}")
    print(f"[+] Num of train malicious samples for client {client_id}: {(y_train==1.0).sum()}")
    print(f"[+] Num of features for client {client_id}: {X_train.shape[1]}")

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    Fire(main)
