from typing import Dict, Optional, Tuple, List
from pathlib import Path

import flwr as fl
import tensorflow as tf
import numpy as np
from utils import get_data, get_model


class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Average thresholds and sum anomalies."""
        if not results:
            return None

        # No need to weigh using the number of examples because they are the same
        # but in general we may want to take this into account
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

    model = get_model()

    # Create custom strategy that aggregates client metrics
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
    x_train, x_test = get_data()

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss = model.evaluate(x_test, x_test)

        # Ideally the threshold should be averaged over all the thresholds instead
        # but I don't know how to do that easily.
        # Instead look at the aggregate client results for the correct metrics.
        x_train_pred = model.predict(x_train)
        train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

        # Get reconstruction loss threshold from training data.
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