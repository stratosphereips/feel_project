from typing import Dict, Optional, Tuple, List
from pathlib import Path

import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        threshold = np.mean([r.metrics["threshold"]  for _, r in results])
        anomalies = np.sum([r.metrics["anomalies"] for _, r in results])
        print(f"Round {rnd} threshold averaged from client results: {threshold}")
        print(f"Round {rnd} total number of anomalies from client results: {anomalies}")

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)



def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    TIME_STEPS = 288

    model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(TIME_STEPS, 1)),
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


    # Create strategy


    strategy = AggregateCustomMetricStrategy(
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=3,
        min_eval_clients=1,
        min_available_clients=5,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": 10},
        strategy=strategy,
        certificates=(
            Path(".cache/certificates/ca.crt").read_bytes(),
            Path(".cache/certificates/server.pem").read_bytes(),
            Path(".cache/certificates/server.key").read_bytes(),
        ),
    )


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
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

    # Use the last 5k training examples as a validation set
    # x_val, y_val = x_train[45000:50000], y_train[45000:50000]

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss = model.evaluate(x_test, x_test)

        # Calculate the threshold
        # Ideally the threshold should be averaged over all the thresholds instead
        x_train_pred = model.predict(x_train)
        train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

        # Get reconstruction loss threshold.
        threshold = np.max(train_mae_loss)

        x_test_pred = model.predict(x_test)
        test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
        test_mae_loss = test_mae_loss.reshape((-1))

        # Detect all the samples which are anomalies.
        anomalies = test_mae_loss > threshold

        return loss, {"threshold": threshold, "anomalies": np.sum(anomalies)}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()