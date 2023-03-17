import pickle
import warnings
from collections import defaultdict

from flwr.common import Metrics
from sklearn.metrics import classification_report

from common.CustomFedAdam import CustomFedAdam

warnings.filterwarnings("ignore")

from typing import Dict, Optional, Tuple, List
from pathlib import Path

import flwr as fl
import tensorflow as tf
import numpy as np
from common.utils import MinMaxScaler, pprint_cm, deserialize, serialize
from common.models import MultiHeadAutoEncoder
from common.data_loading import load_mal_data, load_ben_data, create_supervised_dataset
import pandas as pd
import fire
from common.config import Config


class AggregateCustomMetricStrategy(fl.server.strategy.FedAdam):
    def __init__(self, day, config, **kwds):
        kwds["on_fit_config_fn"] = self.fit_config
        kwds["on_evaluate_config_fn"] = self.evaluate_config
        kwds["evaluate_metrics_aggregation_fn"] = self.evaluate_metrics
        kwds["fit_metrics_aggregation_fn"] = self.evaluate_metrics
        super().__init__(**kwds)
        self.day = day
        self.config = config
        self.threshold = 100
        self.scaler = None
        self.epoch = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Parameters]:
        results = [(proxy, r) for proxy, r in results if r.num_examples != 0]

        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        # if aggregated_weights is not None and server_round == 9:
        #     # Save aggregated_weights on the final round
        #     print(f"[*] Saving round {server_round} aggregated_weights...")
        #     np.savez(f"model-weights.npz", *aggregated_weights)

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
            self.scaler = sum(
                (MinMaxScaler.load(r.metrics["scaler"]) for _, r in results),
                start=MinMaxScaler(),
            )
            with self.config.scaler_file(self.day).open("wb") as fb:
                pickle.dump(self.scaler, fb)

        self.epoch += self.config.local_epochs(server_round)

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

        anomalies_mal = np.sum(
            [r.metrics["anomalies_mal"] for _, r in results if r != None]
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
            f"[*] Round {server_round} accuracy avg from client results: {100 * accuracies:.2f}%"
        )

        # Weigh TPR of each client by number of examples used
        tprs = [r.metrics["tpr"] * r.num_examples for _, r in results if r != None]
        # Aggregate and print custom metric
        tpr_aggregated = 100 * sum(tprs) / sum(examples)
        print(
            f"[*] Round {server_round} TPR weighted avg from client results: {tpr_aggregated:.2f}%"
        )
        tprs = np.mean([r.metrics["tpr"] for _, r in results if r != None])
        print(
            f"[*] Round {server_round} TPR avg from client results: {100 * tprs:.2f}%"
        )

        # Weigh FPR of each client by number of examples used
        fprs = [r.metrics["fpr"] * r.num_examples for _, r in results if r != None]
        # Aggregate and print custom metric
        fpr_aggregated = 100 * sum(fprs) / sum(examples)
        print(
            f"[*] Round {server_round} FPR weighted avg from client results: {fpr_aggregated:.2f}%"
        )
        fprs = np.mean([r.metrics["fpr"] for _, r in results if r != None])
        print(
            f"[*] Round {server_round} FPR avg from client results: {100 * fprs:.2f}%"
        )

        conf_matrices = np.array(
            [deserialize(result.metrics["confusion_matrix"]) for _, result in results]
        )
        conf_matrix = conf_matrices.sum(axis=0)
        pprint_cm(conf_matrix, ["Benign", "Malicious"])

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(server_round, results, failures)

    def fit_config(self, rnd: int):
        """Return training configuration dict for each round.
        Keep batch size fixed at 32, perform two rounds of training with one
        local epoch, increase to two local epochs afterwards.
        """

        config = {
            "start_epoch": self.epoch,
            "local_epochs": self.config.local_epochs(rnd),
            "threshold": self.threshold,
        }
        return config

    def evaluate_config(self, rnd: int):
        """Return evaluation configuration dict for each round.
        Perform five local evaluation steps on each client (i.e., use five
        batches) during rounds one to three, then increase to ten local
        evaluation steps.
        """
        config = {
            "val_steps": 64,
            "threshold": self.threshold,
        }
        if rnd > 1:
            config["scaler"] = serialize(self.scaler)

        return config

    @staticmethod
    def evaluate_metrics(client_metrics: List[Tuple[int, Metrics]]) -> Metrics:
        metrics = {}
        # consistent ordering so that weighted average makes sense
        client_metrics.sort(key=lambda x: x[0])

        client_examples = np.array([num_examples for num_examples, _ in client_metrics])

        metrics_transpose = defaultdict(list)
        for _, m in client_metrics:
            for metric_name, metric_value in m.items():
                metrics_transpose[metric_name].append(metric_value)
        metrics_transpose = {
            metric_name: np.array(metric_values)
            for metric_name, metric_values in metrics_transpose.items()
        }
        ignore_set = {"id", "scaler", "tracker", "confusion_matrix"}
        ignore_set.update([f"global_{x}" for x in ignore_set])

        for metric_name, metric_values in metrics_transpose.items():
            if metric_name in ignore_set:
                continue
            if np.issubdtype(metric_values.dtype, np.integer):
                metrics[metric_name] = metric_values.sum()
            else:
                finite_client_examples = client_examples[np.isfinite(metric_values)]
                metric_values = metric_values[np.isfinite(metric_values)]
                metrics[metric_name] = np.average(
                    metric_values, weights=finite_client_examples
                )

        matrix_metrics = {"tp", "tn", "fp", "fn"}
        for matrix_el in matrix_metrics:
            for _, client_m in client_metrics:
                if matrix_el not in client_m:
                    continue
                metrics[f'{matrix_el}_{client_m["id"]:02}'] = client_m[matrix_el]

        return metrics


def main(day: int, config_path: str = None, **overrides):
    config = Config.load(config_path, **overrides)

    assert config.num_fit_clients <= config.num_evaluate_clients

    tf.keras.utils.set_random_seed(config.seed)
    model = MultiHeadAutoEncoder(config)
    model.compile()
    if config.load_model and day > 1 and config.model_file(day - 1).exists():
        model.load_weights(config.model_file(day - 1))
        num_rounds = config.server.num_rounds_other_days
    else:
        num_rounds = config.server.num_rounds_first_day
    frac_fit = config.num_fit_clients / config.num_evaluate_clients
    assert config.num_fit_clients <= config.num_evaluate_clients
    # Create custom strategy that aggregates client metrics
    strategy = AggregateCustomMetricStrategy(
        day=day,
        config=config,
        fraction_evaluate=1.0,
        min_fit_clients=config.num_evaluate_clients,
        min_evaluate_clients=config.num_evaluate_clients,
        min_available_clients=config.num_evaluate_clients,
        evaluate_fn=get_eval_fn(model, day, config),
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
        eta=config.server.learning_rate,
    )
    # Start Flower server (SSL-enabled) for n rounds of federated learning
    hist = fl.server.start_server(
        server_address=f"{config.ip_address}:{config.port}",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        # certificates=(
        #     Path(".cache/certificates/ca.crt").read_bytes(),
        #     Path(".cache/certificates/server.pem").read_bytes(),
        #     Path(".cache/certificates/server.key").read_bytes(),
        # ),
    )
    model.save_weights(config.model_file(day))

    with config.results_file(day).open("wb") as f:
        pickle.dump(hist, f)


def get_eval_fn(model, day, experiment_config: Config):
    """Return an evaluation function for server-side evaluation."""

    X_test, y_test = load_data(experiment_config.data_dir, day)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        threshold = config["threshold"] if config else 100

        if server_round > 1:
            with experiment_config.scaler_file(day).open("rb") as f:
                scaler = pickle.load(f)
        else:
            scaler = MinMaxScaler().from_min_max(-10 * np.ones(36), 10 * np.ones(36))

        X_test_ = scaler.transform(X_test)

        # Detect all the samples which are anomalies.
        mse, _, _ = model.eval(X_test_, y_test)
        y_pred = (mse > threshold).astype(float).T

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            labels=[0.0, 1.0],
            target_names=["Benign", "Malicious"],
        )

        # Detect all the samples which are anomalies.
        # anomalies_ben = sum(mse_ben > threshold)
        # anomalies_mal = []
        # for folder in list(X_test_mal.keys()):
        #     anomalies_mal.append(sum(mse_mal[folder] > threshold))
        #
        # num_malware = 0
        # for folder in list(X_test_mal.keys()):
        #     num_malware += X_test_mal[folder].shape[0]
        #
        # fp = anomalies_ben
        # tp = sum(anomalies_mal)
        # tn = X_test_.shape[0] - fp
        # fn = num_malware - tp
        #
        # accuracy = (tp + tn) / (tp + tn + fp + fn)
        # # tpr = tp / (tp + fn)
        # # fpr = fp / (fp + tn)
        #
        # tpr = tp / num_malware
        # fpr = fp / X_test_.shape[0]

        return mse.sum(), report
        #        {
        #     "threshold": threshold,
        #     "anomalies_ben": anomalies_ben,
        #     "anomalies_mal": anomalies_mal,
        #     "accuracy": accuracy,
        #     "tpr": tpr,
        #     "fpr": fpr
        # }

    return evaluate


def load_data(data_dir, day):
    ben_train, ben_test = zip(
        *[
            load_ben_data(
                day, client, data_dir, drop_labels=False, drop_four_tuple=True
            )
            for client in range(1, 10)
        ]
    )
    X_ben_test = pd.concat(ben_test, axis=0)
    X_mal_test = pd.concat(
        load_mal_data(
            day + 1, data_dir, drop_labels=False, drop_four_tuple=True
        ).values(),
        axis=0,
    )
    X_test, _, y_test, _ = create_supervised_dataset(X_ben_test, X_mal_test, 0.0)

    return X_test, y_test


if __name__ == "__main__":
    fire.Fire(main)
