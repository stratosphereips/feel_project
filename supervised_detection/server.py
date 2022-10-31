import warnings

warnings.filterwarnings('ignore')

from typing import Dict, Optional, Tuple, List

import flwr as fl
import tensorflow as tf
import numpy as np
from pathlib import Path
from fire import Fire

from common.utils import serialize_array, deserialize_string, client_malware_map, scale_data, MinMaxScaler, serialize, \
    deserialize, pprint_cm
from common.models import get_classification_model
from common.data_loading import load_mal_data, load_ben_data, create_supervised_dataset
import pickle
import pandas as pd
from sklearn.metrics import classification_report
import argparse
import os


class AggregateCustomMetricStrategy(fl.server.strategy.FedAvgM):
    def __init__(self, **kwds):
        super().__init__(**kwds)

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException],
    ) -> Optional[fl.common.Parameters]:
        results = [result for result in results if result[1].num_examples]

        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        if aggregated_weights is not None and server_round == 9:
            # Save aggregated_weights on the final round
            print(f"[*] Saving round {server_round} aggregated_weights...")
            np.savez(f"model-weights.npz", *aggregated_weights)

        if server_round == 1:
            scaler = sum((MinMaxScaler.load(r.metrics['scaler']) for _, r in results), start=MinMaxScaler())
            with open('round-1-scaler.pckl', 'wb') as fb:
                pickle.dump(scaler, fb)

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

        results = [result for result in results if result[1].num_examples]

        anomalies_ben = np.sum([r.metrics["anomalies_ben"] for _, r in results if r != None])
        print(f"[*] Round {server_round} total number of ben. anomalies from client results: {int(anomalies_ben)}")

        anomalies_mal = np.mean([np.sum(r.metrics["anomalies_mal"]) for _, r in results if r != None])
        print(f"[*] Round {server_round} total number of mal. anomalies from client results: {int(anomalies_mal)}")

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results if r != None]
        examples = [r.num_examples for _, r in results if r != None]

        # Aggregate and print custom metric
        accuracy_aggregated = 100 * sum(accuracies) / sum(examples)
        print(f"[*] Round {server_round} accuracy weighted avg from client results: {accuracy_aggregated:.2f}%")

        accuracies = np.mean([r.metrics["accuracy"] for _, r in results if r != None])
        print(f"[*] Round {server_round} accuracy avg from client results: {100 * accuracies:.2f}%")

        # Weigh TPR of each client by number of examples used
        tprs = [r.metrics["tpr"] * r.num_examples for _, r in results if r != None]
        # Aggregate and print custom metric
        tpr_aggregated = 100 * sum(tprs) / sum(examples)
        print(f"[*] Round {server_round} TPR weighted avg from client results: {tpr_aggregated:.2f}%")
        tprs = np.mean([r.metrics["tpr"] for _, r in results if r != None])
        print(f"[*] Round {server_round} TPR avg from client results: {100 * tprs:.2f}%")

        # Weigh FPR of each client by number of examples used
        fprs = [(1 - r.metrics["fpr"]) * r.num_examples for _, r in results if r != None]
        # Aggregate and print custom metric
        fpr_aggregated = 100 * sum(fprs) / sum(examples)
        print(f"[*] Round {server_round} FPR weighted avg from client results: {fpr_aggregated:.2f}%")
        fprs = np.mean([1 - r.metrics["fpr"] for _, r in results if r != None])
        print(f"[*] Round {server_round} FPR avg from client results: {100 * fprs:.2f}%")

        conf_matrices = np.array([deserialize(result.metrics['confusion_matrix']) for _, result in results])
        conf_matrix = conf_matrices.sum(axis=0)
        pprint_cm(conf_matrix, ['Benign', 'Malicious'])

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(server_round, results, failures)


def main(day: int, model_path: str, seed: int = 8181, data_dir: str = '../data', num_clients: int = 3) -> None:
    """
    Flower server
    @param seed: Random seed
    @param data_dir: Path to the data directory
    @param num_clients: Number of clients
    @param model_path Path to the autoencoder model
    """
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation

    model_path = Path(model_path)
    data_dir = Path(data_dir)

    tf.keras.utils.set_random_seed(seed)

    model = get_classification_model(model_path)
    num_rounds = 50

    print(f"{num_clients=}")

    # Create custom strategy that aggregates client metrics
    strategy = AggregateCustomMetricStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=get_eval_fn(model, day, Path(data_dir)),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
        server_momentum=2
    )

    # Start Flower server (SSL-enabled) for n rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8000",
        config=fl.server.ServerConfig(num_rounds=num_rounds, round_timeout=10.0),
        strategy=strategy,
        # certificates=(
        #    Path(".cache/certificates/ca.crt").read_bytes(),
        #    Path(".cache/certificates/server.pem").read_bytes(),
        #    Path(".cache/certificates/server.key").read_bytes(),
        # ),
    )

    # model.save(f'day{day}_{seed}_model.h5')
    tf.keras.models.save_model(model, f'models/day{day}_{seed}_model')


def get_eval_fn(model, day, data_dir):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    # X_train = pd.DataFrame()
    X_test_ben = pd.DataFrame()
    X_test_mal = pd.DataFrame()
    for client_id in range(1, 11):
        _, test_temp = load_ben_data(day, client_id, data_dir, drop_labels=False)
        mal_temp = load_mal_data(day + 1, data_dir, drop_labels=False)[client_malware_map[client_id]]
        # X_train = pd.concat([X_train, train_temp], ignore_index=True)
        X_test_ben = pd.concat([X_test_ben, test_temp], ignore_index=True)
        X_test_mal = pd.concat([X_test_mal, mal_temp], ignore_index=True)

    X_test,_, y_test, _ = create_supervised_dataset(X_test_ben, X_test_mal, 0.0)

    # The `evaluate` function will be called after every round
    def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar]
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters

        # Read the stored per-feature maximums and minimums
        if server_round > 1:
            with open(f'round-1-scaler.pckl', 'rb') as f:
                scaler = pickle.load(f)
        else:
            scaler = MinMaxScaler().from_min_max(-10 * np.ones(36), 10 * np.ones(36))

        X_test_ = scaler.transform(X_test)

        # Detect all the samples which are anomalies.
        prediction = model.predict(X_test_)
        prediction = (prediction > 0.5).astype(float).T[0]
        bce = model.evaluate(X_test_, y_test, 64, steps=None)

        rec_mal = dict()
        mse_mal = dict()

        report = classification_report(y_test, prediction, output_dict=True)
        # Detect all the samples which are anomalies.

        return bce, report

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    mal_data = load_mal_data(5, Path('../data'))["CTU-Malware-Capture-Botnet-67-1"]
    config = {
        "batch_size": 64,
        "local_epochs": 1 if rnd < 2 else 2,
        "neg_dataset": serialize(mal_data)
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    config = {
        "val_steps": 10
    }
    if rnd > 1:
        with open('round-1-scaler.pckl', 'rb') as f:
            scaler = pickle.load(f)
            config['scaler'] = scaler.dump()

    return config


if __name__ == "__main__":
    Fire(main)