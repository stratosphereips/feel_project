import os
import time
import warnings

from flwr.server import ClientManager

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from collections import defaultdict
from common.config import Config, Setting
from typing import Dict, Optional, Tuple, List
import numpy as np
from fire import Fire
import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays, Metrics
from flwr.server.client_proxy import ClientProxy

from common.utils import MinMaxScaler, serialize, deserialize, pprint_cm
from common.models import MultiHeadAutoEncoder, SimpleClassifier
from common.data_loading import (
    load_mal_data,
    load_ben_data,
    create_supervised_dataset,
    load_vaccine_data,
    load_evaluation_data,
    load_multiple
)
import mlflow


class AggregateCustomMetricStrategy(fl.server.strategy.FedAdam):
    def __init__(self, day, config, **kwds):
        kwds["on_fit_config_fn"] = self.fit_config
        kwds["on_evaluate_config_fn"] = self.evaluate_config
        kwds["evaluate_metrics_aggregation_fn"] = self.evaluate_metrics
        kwds["fit_metrics_aggregation_fn"] = self.evaluate_metrics
        super().__init__(**kwds)
        self.day = day
        self.config = config
        self.prev_median_loss = np.inf
        self.scaler = None
        self.epoch = 0
        self.best_val_params = None
        self.best_round = -1
        self.best_val_loss = np.inf
        self.best_val_acc = -np.inf
        self.best_val_auc = -np.inf
        self.val_losses = []
        self.val_acc = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        results = [(proxy, r) for proxy, r in results if r.num_examples != 0]
        results = sorted(results, key=lambda result: result[1].metrics["id"])
        if server_round == 1:
            self.scaler = sum(
                (MinMaxScaler.load(r.metrics["scaler"]) for _, r in results),
                start=MinMaxScaler(),
            )
            with self.config.scaler_file(self.day).open("wb") as fb:
                pickle.dump(self.scaler, fb)

        classification_losses = np.array([r.metrics['val_loss'] for _, r in results])
        #prox_losses = np.array([r.metrics['val_prox_loss'] for _, r in results])
        print(f'\nClassification losses after round {server_round}: \n{classification_losses}\n'
              f'median: {np.median(classification_losses)}\t '
              f'max is client {np.argmax(classification_losses)}: {np.max(classification_losses)}\n')

        # print(
        #     f"\nProximal losses after round {server_round}: \n{prox_losses}\n"
        #     f"median: {np.median(prox_losses)}\t "
        #     f"max is client {np.argmax(prox_losses)}: {np.max(prox_losses)}\n"
        # )

        n_examples = sum((r.num_examples for _, r in results))
        val_accs = np.array([r.metrics["val_acc"] * r.num_examples for _, r in results])
        val_acc = val_accs.sum() / n_examples

        val_loss = (
            np.array([r.metrics["val_loss"] * r.num_examples for _, r in results]).sum()
            / n_examples
        )

        val_aucs = np.array([r.metrics["val_acc"] * r.num_examples for _, r in results])
        val_auc = val_accs.sum() / n_examples

        self.val_acc.append((server_round, val_acc))
        self.val_losses.append((server_round, val_loss))


        print(f"Aggregating {len(results)} results")
        weights, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        print(
            f"[*] Round {server_round} val loss: {val_loss}, val acc: {val_acc}, val auc: {val_auc}"
        )

        if val_acc > self.best_val_acc:
            print(
                "\t\t=============\n"
                f"[*] Round {server_round} best val acc so far: {val_acc}, saving model\n"
                "\t\t============="
            )
            self.best_val_acc = val_acc
            self.best_val_params = parameters_to_ndarrays(weights)
            self.best_val_loss = val_loss
            self.best_round = server_round

        self.epoch += self.config.local_epochs(server_round)

        return weights, metrics_aggregated

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
        # print(
        #     "============================\n   Classification Results   \n============================"
        # )
        cls_fp = np.sum([r.metrics["cls_fp"] for _, r in results if r != None])
        # print(
        #     f"[*] Round {server_round} total number of ben. anomalies from client results: {int(cls_fp)}"
        # )

        cls_tp = np.sum([r.metrics["cls_tp"] for _, r in results if r != None])
        # print(
        #     f"[*] Round {server_round} total number of mal. anomalies from client results: {int(cls_tp)}"
        # )

        # Weigh accuracy of each client by number of examples used
        accuracies = [
            r.metrics["class_accuracy"] * r.num_examples
            for _, r in results
            if r != None
        ]
        examples = [r.num_examples for _, r in results if r != None]

        # Aggregate and print custom metric
        accuracy_aggregated = 100 * sum(accuracies) / sum(examples)
        # print(
        #     f"[*] Round {server_round} accuracy weighted avg from client results: {accuracy_aggregated:.2f}%"
        # )

        accuracies = np.mean(
            [r.metrics["class_accuracy"] for _, r in results if r != None]
        )
        # print(
        #     f"[*] Round {server_round} accuracy avg from client results: {100 * accuracies:.2f}%"
        # )

        # Weigh TPR of each client by number of examples used
        cls_tprs = np.array(
            [r.metrics["cls_tpr"] * r.num_examples for _, r in results if r != None]
        )
        cls_tprs = cls_tprs[~np.isnan(cls_tprs)]
        # Aggregate and print custom metric
        cls_tpr_aggregated = 100 * sum(cls_tprs) / sum(examples)
        # print(
        #     f"[*] Round {server_round} TPR weighted avg from client results: {cls_tpr_aggregated:.2f}%"
        # )
        cls_tprs = np.mean([r.metrics["cls_tpr"] for _, r in results if r != None])
        # print(
        #     f"[*] Round {server_round} TPR avg from client results: {100 * cls_tprs:.2f}%"
        # )

        # Weigh FPR of each client by number of examples used
        cls_fprs = np.array(
            [
                (1 - r.metrics["cls_fpr"]) * r.num_examples
                for _, r in results
                if r != None
            ]
        )
        cls_fprs = cls_fprs[~np.isnan(cls_fprs)]
        # Aggregate and print custom metric
        cls_fpr_aggregated = 100 * sum(cls_fprs) / sum(examples)
        # print(
        #     f"[*] Round {server_round} FPR weighted avg from client results: {cls_fpr_aggregated:.2f}%"
        # )
        cls_fprs = np.mean([1 - r.metrics["cls_fpr"] for _, r in results if r != None])
        # print(
        #     f"[*] Round {server_round} FPR avg from client results: {100 * cls_fprs:.2f}%"
        # )

        if self.config.setting == Setting.LOCAL:
            g_conf_matrices = np.array(
                [
                    deserialize(result.metrics["global_confusion_matrix"])
                    for _, result in results
                ]
            )
            g_conf_matrix = g_conf_matrices.mean(axis=0)
            pprint_cm(g_conf_matrix, ["Benign", "Malicious"])
        else:
            conf_matrices = np.array(
                [
                    deserialize(result.metrics["confusion_matrix"])
                    for _, result in results
                ]
            )
            conf_matrix = conf_matrices.sum(axis=0)
            pprint_cm(conf_matrix, ["Benign", "Malicious"])

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(server_round, results, failures)

    def fit_config(self, rnd: int):
        """ """
        mal_data = load_vaccine_data(self.day, self.config)

        config = {
            "start_epoch": self.epoch,
            "local_epochs": self.config.local_epochs(rnd),
            "mal_dataset": serialize(mal_data),
        }

        return config

    def evaluate_config(self, rnd: int):
        """Return evaluation configuration dict for each round.
        Perform five local evaluation steps on each client (i.e., use five
        batches) during rounds one to three, then increase to ten local
        evaluation steps.
        """
        config = {"val_steps": 64}
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
                metrics[metric_name] = np.average(
                    metric_values, weights=client_examples
                )

        matrix_metrics = {'cls_tp', 'cls_tn', 'cls_fp', 'cls_fn', 'val_loss'}
        for matrix_el in matrix_metrics:
            for _, client_m in client_metrics:
                if matrix_el not in client_m:
                    continue
                metrics[f'{matrix_el}_{client_m["id"]:02}'] = client_m[matrix_el]

        return metrics


def get_eval_fn(model: MultiHeadAutoEncoder, day: int, experiment_config: Config):
    """Return an evaluation function for server-side evaluation."""

    X_test, y_test = load_evaluation_data(experiment_config, day, experiment_config.seed)
    print(f"{experiment_config.holdout(day)=}")
    X_holdout, y_holdout = load_multiple(experiment_config.holdout(day))

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters

        # Read the stored per-feature maximums and minimums
        if server_round > 1:
            with experiment_config.scaler_file(day).open("rb") as f:
                scaler = pickle.load(f)
        else:
            scaler = MinMaxScaler().from_min_max(-10 * np.ones(36), 10 * np.ones(36))


        X_test_ = scaler.transform(X_test)

        # Detect all the samples which are anomalies.
        y_pred, bce = model.eval(X_test_, y_test)

        y_cls_pred = (y_pred > 0.5).astype(float)

        supervised_report = classification_report(
            y_test,
            y_cls_pred,
            output_dict=True,
            labels=[0.0, 1.0],
            target_names=["Benign", "Malicious"],
        )

        report = {f"sup_{key}": val for key, val in supervised_report.items()}
        # Detect all the samples which are anomalies.

        print("Evaluate confusion matrix")
        pprint_cm(confusion_matrix(y_test, y_cls_pred), ["Benign", "Malicious"])
        if X_holdout.size:
            X_holdout_ = scaler.transform(X_holdout)
            y_pred_holdout = model.predict(X_holdout_)[:, -1].numpy()
            y_cls_pred_holdout = (y_pred_holdout > 0.5).astype(float)
            print(f"Detected {y_cls_pred_holdout.sum()} out of {len(y_cls_pred_holdout)} holdout samples. Holdout accuracy: {y_cls_pred_holdout.mean()}")

        return bce, report

    return evaluate


def main(day: int, config_path: str = None, **overrides) -> None:
    """
    Flower server
    @param day: Day of the experiment
    @param config_path: Path to the config file override
    @param overrides: additional config overrides
    """
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    config = Config.load(config_path, **overrides)

    tf.keras.utils.set_random_seed(config.seed)

    if config.model.type == 'SimpleClassifier':
        model = SimpleClassifier(config)
    else:
        model = MultiHeadAutoEncoder(config)

    model.compile()

    if config.load_model and day > 1:
        model.load_weights(config.model_file(day - 1))
        num_rounds = config.server.num_rounds_other_days
    else:
        num_rounds = config.server.num_rounds_first_day

    if config.fit_if_no_malware:
        num_fit_clients = config.num_clients
    else:
        num_fit_clients = (
            max(6, config.num_clients) if day == 1 else max(4, config.num_clients)
        )
    print(f"{num_fit_clients=}")

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
    results = fl.server.start_server(
        server_address=f"{config.ip_address}:{config.port}",
        config=fl.server.ServerConfig(num_rounds=num_rounds, round_timeout=90.0),
        strategy=strategy,
        # certificates=(
        #    Path(".cache/certificates/ca.crt").read_bytes(),
        #    Path(".cache/certificates/server.pem").read_bytes(),
        #    Path(".cache/certificates/server.key").read_bytes(),
        # ),
    )

    X_test, y_test = load_evaluation_data(config, day, config.seed)

    # X_test = strategy.scaler.transform(X_test)
    # plot_embedding(X_test, y_test, model)

    model.set_weights(strategy.best_val_params)

    results.metrics_centralized["best_round"] = strategy.best_round
    results.val_losses_distributed = strategy.val_losses
    results.val_acc_distributed = strategy.val_acc

    model.save_weights(config.model_file(day))
    with config.scaler_file(day).open("wb") as f:
        pickle.dump(strategy.scaler, f)
    with config.results_file(day).open("wb") as f:
        pickle.dump(results, f)



if __name__ == "__main__":
    Fire(main)
