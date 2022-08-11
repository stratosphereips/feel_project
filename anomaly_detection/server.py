import warnings
warnings.filterwarnings('ignore')

from typing import Dict, Optional, Tuple, List
from pathlib import Path

import flwr as fl
import tensorflow as tf
import numpy as np
from utils import get_mal_data, get_ben_data, get_model, get_threshold, serialize_array, deserialize_string, scale_data
import pandas as pd
from sklearn import preprocessing
import argparse

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.threshold = 1000

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

        self.threshold = np.mean([r.metrics["threshold"]  for _, r in results])
        print(f"[*] Round {server_round} threshold averaged from client results: {self.threshold:.5f}")
        # if server_round == 10:
        print(f"[*] Saving round {server_round} average threshold...")
        np.savez(f"round-{server_round}-threshold.npz", threshold=self.threshold)

        if server_round == 1:
            X_min = np.min([deserialize_string(r.metrics["X_min"]) for _, r in results], axis=0)
            print(f"[*] Saving round {server_round} minimums... {X_min}")
            np.savez(f"round-{server_round}-min.npz", X_min=np.array(X_min))

            X_max = np.max([deserialize_string(r.metrics["X_max"]) for _, r in results], axis=0)
            print(f"[*] Saving round {server_round} maximums... {X_max}")
            np.savez(f"round-{server_round}-max.npz", X_max=np.array(X_max))

        return aggregated_weights
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Sum anomalies reported from all clients."""
        if not results:
            return None

        anomalies_ben = np.sum([r.metrics["anomalies_ben"] for _, r in results])
        print(f"[*] Round {server_round} total number of anomalies from client results: {anomalies_ben}")

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
        config=fl.server.ServerConfig(num_rounds=5),
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
    # X_train = pd.DataFrame()
    X_test_ben = pd.DataFrame()
    for client_id in range(1, 11):
        _, test_temp = get_ben_data(day, client_id)
        # X_train = pd.concat([X_train, train_temp], ignore_index=True)
        X_test_ben = pd.concat([X_test_ben, test_temp], ignore_index=True)

    X_test_mal = get_mal_data()

    # How are we scaling these parameters? A global scaler or the local aggregate?
    # scaler = preprocessing.MinMaxScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test_ben = scaler.transform(X_test_ben)

    # for folder in list(X_test_mal.keys()):
    #     X_test_mal[folder] = scaler.transform(X_test_mal[folder])

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, 
        parameters: fl.common.NDArrays, 
        config: Dict[str, fl.common.Scalar]
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss = model.evaluate(X_test_ben, X_test_ben)

        # Read the stored aggregated threshold from the clients
        if server_round > 0:
            with np.load(f'round-{server_round}-threshold.npz') as data:
                threshold = float(data['threshold'])
        else:
            threshold = 100 # dummy value

        # Read the stored per-feature maximums and minimums
        if server_round > 0:
            with np.load(f'round-1-max.npz') as data:
                X_max = data['X_max']

            with np.load(f'round-1-min.npz') as data:
                X_min = data['X_min']
        else:
            X_max = 1000*np.ones(40)
            X_min = -1000*np.zeros(40)
        
        X_test_ben_ = scale_data(X_test_ben, X_min, X_max)

        # Detect all the samples which are anomalies.
        rec_ben = model.predict(X_test_ben_)
        mse_ben = np.mean(np.power(X_test_ben_ - rec_ben, 2), axis=1)

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
    with np.load(f'round-{rnd}-threshold.npz') as data:
        threshold = float(data['threshold'])

    print("[*] Evaluate config with threshold:", threshold)

    # if rnd == 1:
    with np.load(f'round-1-max.npz') as data:
        X_max = serialize_array(data['X_max'])

    with np.load(f'round-1-min.npz') as data:
        X_min = serialize_array(data['X_min'])
    

    print("[*] Evaluate config with threshold:", threshold)

    val_steps = 5 
    # if rnd < 4 else 10
    return {
        "val_steps": val_steps,
        "threshold": threshold,
        "X_min": X_min,
        "X_max": X_max
    }



if __name__ == "__main__":
    main()
