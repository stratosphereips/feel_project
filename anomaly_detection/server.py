import warnings
warnings.filterwarnings('ignore')

from typing import Dict, Optional, Tuple, List
from pathlib import Path

import flwr as fl
import tensorflow as tf
import numpy as np
from utils import get_mal_data, get_ben_data, get_model, get_threshold
import pandas as pd
from sklearn import preprocessing
import argparse

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Parameters]:
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        if aggregated_weights is not None and server_round == 10:
            # Save aggregated_weights on the final round
            print(f"[*] Saving round {server_round} aggregated_weights...")
            np.savez(f"round-{server_round}-weights.npz", *aggregated_weights)
                        
        return aggregated_weights
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Average thresholds and sum anomalies."""
        if not results:
            return None

        # No need to weigh using the number of examples because they are the same
        # but in general we may want to take this into account
        threshold = np.mean([r.metrics["threshold"]  for _, r in results])
        anomalies_ben = np.sum([r.metrics["anomalies_ben"] for _, r in results])
        print(f"[*] Round {server_round} threshold averaged from client results: {threshold:.5f}")
        print(f"[*] Round {server_round} total number of anomalies from client results: {anomalies_ben}")

        # The below is not needed because it is calculated like that in FedAvg
        # # Weigh loss of each client by number of examples used
        # losses = [r.loss * r.num_examples for _, r in results]
        # examples = [r.num_examples for _, r in results]

        # # Aggregate and print custom metric
        # loss_aggregated = sum(losses) / sum(examples)
        # print(f"Round {server_round} weighted loss aggregated from client results: {loss_aggregated}")
        if server_round == 10:
            print(f"[*] Saving round {server_round} average threshold...")
            np.savez(f"round-{server_round}-threshold.npz", threshold)

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(server_round, results, failures)

def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Flower server")
    parser.add_argument("--day", type=int, choices=(range(1,6)), required=True)
    args = parser.parse_args()
    
    day = args.day

    # TODO: initialize the model parameters from the precious day
    model = get_model()

    # Create custom strategy that aggregates client metrics
    strategy = AggregateCustomMetricStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=1,
        min_available_clients=5,
        evaluate_fn=get_eval_fn(model, day),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        certificates=(
            Path(".cache/certificates/ca.crt").read_bytes(),
            Path(".cache/certificates/server.pem").read_bytes(),
            Path(".cache/certificates/server.key").read_bytes(),
        ),
    )


def get_eval_fn(model, day):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    X_train = pd.DataFrame()
    X_test_ben = pd.DataFrame()
    for client_id in range(1, 11):
        train_temp, test_temp = get_ben_data(day, client_id)
        X_train = pd.concat([X_train, train_temp], ignore_index=True)
        X_test_ben = pd.concat([X_test_ben, test_temp], ignore_index=True)

    X_test_mal = get_mal_data()

    # How are we scaling these parameters? A global scaler or the local aggregate?
    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test_ben = scaler.transform(X_test_ben)

    for folder in list(X_test_mal.keys()):
        X_test_mal[folder] = scaler.transform(X_test_mal[folder])

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, 
        parameters: fl.common.NDArrays, 
        config: Dict[str, fl.common.Scalar]
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss = model.evaluate(X_test_ben, X_test_ben)

        # Ideally the threshold should be averaged over all the thresholds instead
        # but I don't know how to do that easily.
        # Instead look at the aggregate client results for the correct metrics.
        rec = model.predict(X_train)
        mse = np.mean(np.power(X_train - rec, 2), axis=1)
        threshold = get_threshold(X_train, mse)

        # Detect all the samples which are anomalies.
        rec_ben = model.predict(X_test_ben)
        mse_ben = np.mean(np.power(X_test_ben - rec_ben, 2), axis=1)

        rec_mal = dict()
        mse_mal = dict()
        for folder in list(X_test_mal.keys()):
            rec_mal[folder] = model.predict(X_test_mal[folder])
            mse_mal[folder] = np.mean(np.power(X_test_mal[folder] - rec_mal[folder], 2), axis=1)

        # Detect all the samples which are anomalies.
        anomalies_ben = sum(mse_ben > threshold)
        anomalies_mal = []
        for folder in list(X_test_mal.keys()):
            anomalies_mal.append(sum(mse_mal[folder] > threshold))

        return loss, {"threshold": threshold, "anomalies_ben": anomalies_ben, "anomalies_mal": anomalies_mal}

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
