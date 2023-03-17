from gc import callbacks
import warnings

import fire
from sklearn.metrics import confusion_matrix

from common.config import Config, Setting

warnings.filterwarnings("ignore")

from pathlib import Path

import tensorflow as tf
import numpy as np

import flwr as fl
from flwr.common.logger import logger
from common.utils import get_threshold, MinMaxScaler, serialize, client_malware_map
from common.data_loading import (
    load_mal_data,
    load_ben_data,
    load_day_dataset,
    load_centralized_data,
)
from common.models import MultiHeadAutoEncoder
from sklearn.model_selection import train_test_split
import os

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class ADClient(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        X_train,
        X_test_ben,
        X_test_mal,
        client_id,
        config: Config,
        X_test_ben_global=None,
        X_test_mal_global=None,
    ):
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
        self.X_test_ben_global = X_test_ben_global
        self.X_test_mal_global = X_test_mal_global
        self.last_val_loss = np.nan

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        if self.client_id > self.config.num_fit_clients:
            return [], 0, {}

        # Update local model parameters
        if not self.config.setting == Setting.FEDERATED:
            logger.info("Setting weights")
            self.model.set_weights(parameters)
        else:
            logger.info("Training locally only, ignoring new weights")

        # Get hyperparameters for this round
        batch_size: int = self.config.client.batch_size
        start_epoch: int = config["start_epoch"]
        epochs: int = config["local_epochs"]

        X_train = self.scaler.transform(self.X_train)
        X_train, X_val = train_test_split(
            X_train, test_size=self.config.client.val_ratio, random_state=self.seed
        )
        X_train_l = np.hstack([X_train, np.zeros((X_train.shape[0], 1))])
        X_val_l = np.hstack([X_val, np.zeros((X_val.shape[0], 1))])

        tf.summary.trace_on(graph=True, profiler=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=Path(f"../logs/fit/client{self.client_id}")
        )

        # Train the model using hyperparameters from config
        history = self.model.fit(
            X_train_l,
            X_train_l,
            validation_data=(X_val_l, X_val_l),
            batch_size=batch_size,
            initial_epoch=start_epoch,
            epochs=start_epoch + epochs,
            callbacks=[tensorboard_callback],
        )

        loss = history.history["total_loss"][0]
        val_loss = history.history["val_total_loss"][0]
        self.last_val_loss = val_loss

        if np.isnan(loss) or np.isnan(val_loss):
            print("[+] HERE!", np.isnan(loss), np.isnan(val_loss))
            print("[+] HERE!", val_loss, self.max_loss)
            self.model.set_weights(parameters)

        # Calculate the threshold based on the local validation data
        rec = self.model.predict(X_val)[:, :-1]
        mse = np.mean(np.power(X_val - rec, 2), axis=1)
        self.threshold = get_threshold(X_val, mse)

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(X_train)
        if self.config.setting == Setting.LOCAL:
            parameters_prime = [
                np.zeros_like(layer) for layer in self.model.get_weights()
            ]
        results = {
            "id": self.client_id,
            "loss": loss,
            # "accuracy": history.history["accuracy"][0],
            "val_loss": val_loss,
            # "val_accuracy": history.history["val_accuracy"][0],
            "threshold": float(self.threshold),
            "id": self.client_id,
            "scaler": self.scaler.dump(),
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        if not self.config.setting == Setting.FEDERATED:
            logger.info("Setting weights")
            self.model.set_weights(parameters)
        else:
            logger.info("Training locally only, ignoring new weights")

        if "scaler" in config:
            self.scaler = MinMaxScaler.load(config["scaler"])

        print("[+] Received global threshold:", config["threshold"])
        self.threshold = config["threshold"]

        num_examples = len(self.X_test_ben) + len(self.X_test_mal)

        eval_results = self.eval_test(self.X_test_ben, self.X_test_mal)
        eval_results["val_loss"] = self.last_val_loss

        if self.config.setting == Setting.LOCAL:
            global_results = self.eval_test(
                self.X_test_ben_global, self.X_test_mal_global, "global_"
            )
            eval_results.update(global_results)

        return 0.1, int(num_examples), eval_results

    def eval_test(self, X_test_ben, X_test_mal, res_prefix=""):
        X_test_ben_ = self.scaler.transform(X_test_ben)
        num_examples_test = len(X_test_ben_)
        rec_ben = self.model.predict(X_test_ben_)[:, :-1]
        mse_ben = np.mean(np.power(X_test_ben_ - rec_ben, 2), axis=1)
        y_ben = mse_ben > self.threshold
        anomalies_ben = sum(y_ben)
        num_malware = X_test_mal.shape[0]
        anomalies_mal = 0
        y_pred = y_ben.tolist()
        y_true = np.zeros_like(y_ben).tolist()
        if num_malware > 0:
            X_test_mal_ = self.scaler.transform(X_test_mal).astype("float32")
            rec_mal = self.model.predict(X_test_mal_)[:, :-1]
            mse_mal = np.mean(np.power(X_test_mal_ - rec_mal, 2), axis=1)

            y_mal = mse_mal > self.threshold
            anomalies_mal += sum(y_mal)
            y_pred += y_mal.tolist()
            y_true += np.ones_like(y_mal).tolist()
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        fp = int(anomalies_ben)
        tp = int(anomalies_mal)
        tn = num_examples_test - fp
        fn = num_malware - tp
        accuracy = (tp + tn) / (num_examples_test + num_malware)
        tpr = (tp / num_malware) if num_malware else 1.0
        fpr = fp / num_examples_test
        conf_matrix = confusion_matrix(y_true, y_pred)
        if conf_matrix.size == 1:
            conf_matrix = np.array([[conf_matrix[0][0], 0], [0, 0]])
        eval_results = {
            "id": self.client_id,
            "anomalies_ben": int(anomalies_ben),
            "anomalies_mal": int(anomalies_mal),
            "accuracy": (y_true == y_pred).mean(),
            "tpr": tpr,
            "fpr": fpr,
            "confusion_matrix": serialize(conf_matrix),
            "fp": fp,
            "tp": tp,
            "tn": tn,
            "fn": fn,
        }
        return {f"{res_prefix}{key}": value for key, value in eval_results.items()}


def main(client_id: int, day: int, config_path: str = None, **overrides) -> None:
    config = Config.load(config_path, **overrides)
    # Parse command line arguments
    tf.keras.utils.set_random_seed(config.seed)

    # Load and compile Keras model
    model = MultiHeadAutoEncoder(config)
    model.compile()

    X_train, X_test_ben, X_test_mal = load_partition(day, client_id, config)

    if config.setting == Setting.LOCAL:
        X_test_ben_globa, X_test_mal_global = load_global(day, config)
    else:
        X_test_ben_globa = X_test_mal_global = None

    # Start Flower client
    client = ADClient(
        model,
        X_train,
        X_test_ben,
        X_test_mal,
        client_id,
        config,
        X_test_ben_globa,
        X_test_mal_global,
    )

    fl.client.start_numpy_client(
        server_address=f"{config.ip_address}:{config.port}",
        client=client,
        # root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


def load_partition(day: int, client_id: int, config: Config):
    """Load the training and test data to simulate a partition."""
    assert client_id in range(1, 11)
    assert day in range(1, 6)

    X_train, X_test_ben = load_ben_data(day, client_id, config.data_dir)
    X_test_mal = load_day_dataset(
        config.client_malware(client_id, day + 1), drop_labels=True
    )
    print(f"[+] Num train samples for client{client_id}: {X_train.shape[0]}")
    print(f"[+] Num of features for client{client_id}: {X_train.shape[1]}")
    return X_train.values, X_test_ben.values, X_test_mal.values


def load_global(day, config):
    _, _, global_X_test, _, _, global_y_test = load_centralized_data(day, config)
    X_ben = global_X_test[global_y_test == 0]
    X_mal = global_X_test[global_y_test == 1]
    return X_ben, X_mal


if __name__ == "__main__":
    fire.Fire(main)
