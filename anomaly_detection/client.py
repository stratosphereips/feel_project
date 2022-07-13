import argparse
import os
from pathlib import Path

import tensorflow as tf
import pandas as pd
import numpy as np

import flwr as fl

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class ADClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, x_test):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test

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
            self.x_train,
            self.x_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            # "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            # "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss = self.model.evaluate(self.x_test, self.x_test, 32, steps=steps)
        num_examples_test = len(self.x_test)

        x_train_pred = self.model.predict(self.x_train)
        train_mae_loss = np.mean(np.abs(x_train_pred - self.x_train), axis=1)

        # Get reconstruction loss threshold.
        threshold = np.max(train_mae_loss)

        x_test_pred = self.model.predict(self.x_test)
        test_mae_loss = np.mean(np.abs(x_test_pred - self.x_test), axis=1)
        test_mae_loss = test_mae_loss.reshape((-1))

        # Detect all the samples which are anomalies.
        anomalies = test_mae_loss > threshold

        return loss, num_examples_test, {"threshold": threshold, "anomalies": int(np.sum(anomalies))}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, 10), required=True)
    args = parser.parse_args()

    # Load and compile Keras model
    model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(288, 1)),
        tf.keras.layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        tf.keras.layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")


    x_train, x_test = load_partition(args.partition)

    # Start Flower client
    client = ADClient(model, x_train, x_test)

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client,
        root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


def load_partition(idx: int):
    """Load 1/5th of the training and test data to simulate a partition."""
    assert idx in range(5)

    master_url_root = "https://raw.githubusercontent.com/numenta/NAB/master/data/"

    df_small_noise_url_suffix = "artificialNoAnomaly/art_daily_small_noise.csv"
    df_small_noise_url = master_url_root + df_small_noise_url_suffix
    df_small_noise = pd.read_csv(
        df_small_noise_url, parse_dates=True, index_col="timestamp"
    )

    df_daily_jumpsup_url_suffix = "artificialWithAnomaly/art_daily_jumpsup.csv"
    df_daily_jumpsup_url = master_url_root + df_daily_jumpsup_url_suffix
    df_daily_jumpsup = pd.read_csv(
        df_daily_jumpsup_url, parse_dates=True, index_col="timestamp"
    )

    training_mean = df_small_noise.mean()
    training_std = df_small_noise.std()
    df_training_value = (df_small_noise - training_mean) / training_std

    TIME_STEPS = 288

    # Generated training sequences for use in the model.
    def create_sequences(values, time_steps=TIME_STEPS):
        output = []
        for i in range(len(values) - time_steps + 1):
            output.append(values[i : (i + time_steps)])
        return np.stack(output)


    x_train = create_sequences(df_training_value.values)

    # Prepare test data
    df_test_value = (df_daily_jumpsup - training_mean) / training_std

    # Create sequences from test values.
    x_test = create_sequences(df_test_value.values)
    
    return x_train[idx * 749 : (idx + 1) * 749], x_test[idx * 749 : (idx + 1) * 749]


if __name__ == "__main__":
    main()