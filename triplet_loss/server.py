# SPDX-FileCopyrightText: 2021 Sebastian Garcia <sebastian.garcia@agents.fel.cvut.cz>
#  SPDX-License-Identifier: GPL-2.0-only
import warnings

warnings.filterwarnings("ignore")

from typing import Dict, Optional, Tuple, List

import flwr as fl
import tensorflow as tf
import numpy as np
from pathlib import Path
from common.utils import serialize_array, deserialize_string, scale_data
from common.models import get_triplet_loss_model
from common.data_loading import load_mal_data, load_ben_data
import pandas as pd
import argparse
import os


class AggregateCustomMetricStrategy(fl.server.strategy.FedAvgM):
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
        if aggregated_weights is not None and server_round == 9:
            # Save aggregated_weights on the final round
            print(f"[*] Saving round {server_round} aggregated_weights...")
            np.savez(f"model-weights.npz", *aggregated_weights)

        self.threshold = np.mean([r.metrics["threshold"] for _, r in results])
        print(
            f"[*] Round {server_round} threshold averaged from client results: {self.threshold:.5f}"
        )
        # if server_round == 10:

        # # Weigh threshold of each client by number of examples used
        ths = [r.metrics["threshold"] * r.num_examples for _, r in results if r != None]
        examples = [r.num_examples for _, r in results if r != None]

        # # Aggregate and print custom metric
        weighted_th = sum(ths) / sum(examples)
        print(
            f"[*] Round {server_round} threshold weighted avg from client results: {weighted_th:.5f}"
        )

        self.threshold = weighted_th

        print(f"[*] Saving round {server_round} average threshold...")
        np.savez(f"round-{server_round}-threshold.npz", threshold=self.threshold)

        if server_round == 1:
            X_min = np.min(
                [deserialize_string(r.metrics["X_min"]) for _, r in results], axis=0
            )
            print(f"[*] Saving round {server_round} minimums... {X_min}")
            np.savez(f"round-{server_round}-min.npz", X_min=np.array(X_min))

            X_max = np.max(
                [deserialize_string(r.metrics["X_max"]) for _, r in results], axis=0
            )
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

        anomalies_ben = np.sum(
            [r.metrics["anomalies_ben"] for _, r in results if r != None]
        )
        print(
            f"[*] Round {server_round} total number of ben. anomalies from client results: {int(anomalies_ben)}"
        )

        anomalies_mal = np.mean(
            [np.sum(r.metrics["anomalies_mal"]) for _, r in results if r != None]
        )
        print(
            f"[*] Round {server_round} total number of mal. anomalies from client results: {int(anomalies_mal)}"
        )

        # Weigh accuracy of each client by number of examples used
        accuracies = [
            r.metrics["accuracy"] * r.num_examples for _, r in results if r != None
        ]
        examples = [r.num_examples for _, r in results if r != None]

        # Aggregate and print custom metric
        accuracy_aggregated = 100 * sum(accuracies) / sum(examples)
        print(
            f"[*] Round {server_round} accuracy weighted avg from client results: {accuracy_aggregated:.2f}%"
        )

        accuracies = np.mean([r.metrics["accuracy"] for _, r in results if r != None])
        print(
            f"[*] Round {server_round} accuracy avg from client results: {100*accuracies:.2f}%"
        )

        # Weigh TPR of each client by number of examples used
        tprs = [r.metrics["tpr"] * r.num_examples for _, r in results if r != None]
        # Aggregate and print custom metric
        tpr_aggregated = 100 * sum(tprs) / sum(examples)
        print(
            f"[*] Round {server_round} TPR weighted avg from client results: {tpr_aggregated:.2f}%"
        )
        tprs = np.mean([r.metrics["tpr"] for _, r in results if r != None])
        print(f"[*] Round {server_round} TPR avg from client results: {100*tprs:.2f}%")

        # Weigh FPR of each client by number of examples used
        fprs = [r.metrics["fpr"] * r.num_examples for _, r in results if r != None]
        # Aggregate and print custom metric
        fpr_aggregated = 100 * sum(fprs) / sum(examples)
        print(
            f"[*] Round {server_round} FPR weighted avg from client results: {fpr_aggregated:.2f}%"
        )
        fprs = np.mean([r.metrics["fpr"] for _, r in results if r != None])
        print(f"[*] Round {server_round} FPR avg from client results: {100*fprs:.2f}%")

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(server_round, results, failures)


def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Flower server")
    parser.add_argument(
        "--day", type=int, choices=(range(1, 6)), required=True, help="Training day"
    )
    parser.add_argument(
        "--seed", type=int, required=False, default=8181, help="Random seed"
    )
    parser.add_argument(
        "--load",
        type=int,
        choices=(0, 1),
        required=False,
        default=0,
        help="Load a model from disk or not",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        default="/data",
        help="Path to the data direcotry",
    )
    parser.add_argument(
        "--num_clients", type=int, choices=(range(1, 11)), required=False, default=10
    )

    args = parser.parse_args()

    day = args.day
    tf.keras.utils.set_random_seed(args.seed)

    if args.load and day > 1 and os.path.exists(f"models/day{day-1}_{args.seed}_model"):
        model = tf.keras.models.load_model(f"models/day{day-1}_{args.seed}_model")
        num_rounds = 2
    else:
        model = get_triplet_loss_model()
        num_rounds = 10

    frac_fit = args.num_clients / 10

    # Create custom strategy that aggregates client metrics
    strategy = AggregateCustomMetricStrategy(
        fraction_fit=frac_fit,
        fraction_evaluate=1.0,
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=10,
        evaluate_fn=get_eval_fn(model, day, Path(args.data_dir)),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
        server_momentum=0.2,
    )

    # Start Flower server (SSL-enabled) for n rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8000",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        # certificates=(
        #    Path(".cache/certificates/ca.crt").read_bytes(),
        #    Path(".cache/certificates/server.pem").read_bytes(),
        #    Path(".cache/certificates/server.key").read_bytes(),
        # ),
    )

    # model.save(f'day{day}_{args.seed}_model.h5')
    tf.keras.models.save_model(model, f"models/day{day}_{args.seed}_model")


def get_eval_fn(model, day, data_dir):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    # X_train = pd.DataFrame()
    X_test_ben = pd.DataFrame()
    for client_id in range(1, 11):
        _, test_temp = load_ben_data(day, client_id, data_dir)
        # X_train = pd.concat([X_train, train_temp], ignore_index=True)
        X_test_ben = pd.concat([X_test_ben, test_temp], ignore_index=True)

    X_test_mal = load_mal_data(1, data_dir)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss = model.evaluate((X_test_ben, X_test_ben, X_test_ben), X_test_ben)

        # Read the stored aggregated threshold from the clients
        if server_round > 0:
            with np.load(f"round-{server_round}-threshold.npz") as data:
                threshold = float(data["threshold"])
        else:
            threshold = 100  # dummy value

        # Read the stored per-feature maximums and minimums
        if server_round > 0:
            with np.load(f"round-1-max.npz") as data:
                X_max = data["X_max"]

            with np.load(f"round-1-min.npz") as data:
                X_min = data["X_min"]
        else:
            X_max = 10 * np.ones(36)
            X_min = -10 * np.ones(36)

        X_test_ben_ = scale_data(X_test_ben, X_min, X_max)

        # Detect all the samples which are anomalies.
        rec_ben = model.predict((X_test_ben_, X_test_ben_, X_test_ben_))
        mse_ben = np.mean(np.power(X_test_ben_ - rec_ben, 2), axis=1)

        rec_mal = dict()
        mse_mal = dict()
        for folder in list(X_test_mal.keys()):
            X_test_mal_ = scale_data(X_test_mal[folder], X_min, X_max)
            rec_mal[folder] = model.predict((X_test_mal_, X_test_mal_, X_test_mal_))
            mse_mal[folder] = np.mean(
                np.power(X_test_mal_ - rec_mal[folder], 2), axis=1
            )

        # Detect all the samples which are anomalies.
        anomalies_ben = sum(mse_ben > threshold)
        anomalies_mal = []
        for folder in list(X_test_mal.keys()):
            anomalies_mal.append(sum(mse_mal[folder] > threshold))

        num_malware = 0
        for folder in list(X_test_mal.keys()):
            num_malware += X_test_mal[folder].shape[0]

        fp = anomalies_ben
        tp = sum(anomalies_mal)
        tn = X_test_ben_.shape[0] - fp
        fn = num_malware - tp

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        # tpr = tp / (tp + fn)
        # fpr = fp / (fp + tn)

        tpr = tp / num_malware
        fpr = fp / X_test_ben_.shape[0]

        return loss, {
            "threshold": threshold,
            "anomalies_ben": anomalies_ben,
            "anomalies_mal": anomalies_mal,
            "accuracy": accuracy,
            "tpr": tpr,
            "fpr": fpr,
        }

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 64,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    with np.load(f"round-{rnd}-threshold.npz") as data:
        threshold = float(data["threshold"])

    print("[*] Evaluate config with threshold:", threshold)

    # if rnd == 1:
    with np.load(f"round-1-max.npz") as data:
        X_max = serialize_array(data["X_max"])

    with np.load(f"round-1-min.npz") as data:
        X_min = serialize_array(data["X_min"])

    print("[*] Evaluate config with threshold:", threshold)

    val_steps = 10
    # if rnd < 4 else 10
    return {
        "val_steps": val_steps,
        "threshold": threshold,
        "X_min": X_min,
        "X_max": X_max,
    }


if __name__ == "__main__":
    main()
