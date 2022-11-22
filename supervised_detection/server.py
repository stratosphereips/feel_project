import inspect
import os
import types
from collections import defaultdict
from copy import deepcopy

from common.CustomFedAdam import CustomFedAdam
from common.config import Config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings

warnings.filterwarnings('ignore')

from functools import reduce

from flwr.common import FitRes, Parameters, Scalar, NDArrays, parameters_to_ndarrays, ndarrays_to_parameters, Metrics
from flwr.server.client_proxy import ClientProxy

from typing import Dict, Optional, Tuple, List, Union

import flwr as fl
import numpy as np
from pathlib import Path
from fire import Fire

from common.utils import serialize_array, deserialize_string, client_malware_map, scale_data, MinMaxScaler, serialize, \
    deserialize, pprint_cm, plot_embedding
from common.models import get_classification_model, MultiHeadAutoEncoder, MetricsTracker
from common.data_loading import load_mal_data, load_ben_data, create_supervised_dataset, load_day_dataset
import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf


class AggregateCustomMetricStrategy(CustomFedAdam):
    def __init__(self, day, config, n_cls_layers, **kwds):
        kwds['on_fit_config_fn'] = self.fit_config
        kwds['on_evaluate_config_fn'] = self.evaluate_config
        kwds['evaluate_metrics_aggregation_fn'] = self.evaluate_metrics
        kwds['fit_metrics_aggregation_fn'] = self.evaluate_metrics
        super().__init__(n_cls_layers, **kwds)
        self.day = day
        self.config = config
        self.prev_median_loss = np.inf
        self.proxy_spheres = {}
        self.scaler = None
        self.epoch = 0
        self.threshold = 100
        self.best_val_params = None
        self.best_round = -1
        self.best_val_acc = -np.inf
        self.val_losses = []
        self.val_acc = []

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException],
    ) -> Optional[fl.common.Parameters]:
        # results = [result for result in results if result[1].num_examples]
        results = sorted(results, key=lambda result: result[1].metrics['id'])
        if server_round == 1:
            self.scaler = sum((MinMaxScaler.load(r.metrics['scaler']) for _, r in results), start=MinMaxScaler())
            with self.config.scaler_file(self.day).open('wb') as fb:
                pickle.dump(self.scaler, fb)

        classification_losses = np.array([r.metrics['val_classification_loss'] for _, r in results])
        reconstruction_losses = np.array([r.metrics['val_reconstruction_loss'] for _, r in results])
        prox_losses = np.array([r.metrics['val_prox_loss'] for _, r in results])
        print(f'\nClassification losses after round {server_round}: \n{classification_losses}\n'
              f'median: {np.median(classification_losses)}\t '
              f'max is client {np.argmax(classification_losses)}: {np.max(classification_losses)}\n')

        print(f'\nReconstruction losses after round {server_round}: \n{reconstruction_losses}\n'
              f'median: {np.median(reconstruction_losses)}\t '
              f'max is client {np.argmax(reconstruction_losses)}: {np.max(reconstruction_losses)}\n')

        print(f'\nProximal losses after round {server_round}: \n{prox_losses}\n'
              f'median: {np.median(prox_losses)}\t '
              f'max is client {np.argmax(prox_losses)}: {np.max(prox_losses)}\n')

        n_examples = sum((r.num_examples for _, r in results))
        val_accs = np.array([r.metrics['val_acc'] * r.num_examples for _, r in results])
        val_acc = val_accs.sum() / n_examples

        val_loss = np.array([r.metrics['val_acc'] * r.num_examples for _, r in results]).sum() / n_examples

        self.val_acc.append((server_round, val_acc))
        self.val_losses.append((server_round, val_loss))

        # tracker = sum((MetricsTracker.deserialize(r.metrics['tracker']) for _, r in results), MetricsTracker())
        # for t in tracker.get_trackers():
        #     print(f"{t.name}: {t.result().numpy()}")

        #
        # total_losses = np.array([r.metrics['val_loss'] for _, r in results])
        # threshold = np.median(total_losses) * 100
        # print(f'Not aggregating results of {(total_losses > threshold).nonzero()} clients, because they are diverging')
        # results = [(proxy, r) for proxy, r in results if r.metrics['val_loss'] < threshold]
        #
        # threshold_previous = self.prev_median_loss * 100
        # print(f'Not aggregating results of {(total_losses > threshold_previous).nonzero()} clients, '
        #       f'because they values diverge form last round too much\n\n')
        # results = [(proxy, r) for proxy, r in results if r.metrics['val_loss'] < threshold_previous]
        # self.prev_median_loss = min(
        #     self.prev_median_loss,
        #     np.median(np.array([r.metrics['val_loss'] for _, r in results]))
        # )

        ths = [r.metrics["threshold"] * r.num_examples for _, r in results if r != None]
        examples = [r.num_examples for _, r in results if r != None]

        # # Aggregate and print custom metric
        weighted_th = sum(ths) / sum(examples)
        print(f"[*] Round {server_round} threshold weighted avg from client results: {weighted_th:.5f}")

        self.threshold = weighted_th

        self.proxy_spheres = {r.metrics['id']: deserialize(r.metrics['proxy_spheres']) for _, r in results}

        print(f'Aggregating {len(results)} results')
        aggregated_weights = super().aggregate_fit(server_round, results, failures)

        if val_acc >= self.best_val_acc:
            print(f"[*] Round {server_round} best val accuracy so far: {val_acc}, saving model")
            self.best_val_params = deepcopy(self.current_weights)
            self.best_val_acc = val_acc
            self.best_round = server_round

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

        results = [result for result in results if result[1].num_examples]
        print("============================\n   Classification Results   \n============================")
        cls_fp = np.sum([r.metrics["cls_fp"] for _, r in results if r != None])
        print(f"[*] Round {server_round} total number of ben. anomalies from client results: {int(cls_fp)}")

        cls_tp = np.sum([r.metrics["cls_tp"] for _, r in results if r != None])
        print(f"[*] Round {server_round} total number of mal. anomalies from client results: {int(cls_tp)}")

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["class_accuracy"] * r.num_examples for _, r in results if r != None]
        examples = [r.num_examples for _, r in results if r != None]

        # Aggregate and print custom metric
        accuracy_aggregated = 100 * sum(accuracies) / sum(examples)
        print(f"[*] Round {server_round} accuracy weighted avg from client results: {accuracy_aggregated:.2f}%")

        accuracies = np.mean([r.metrics["class_accuracy"] for _, r in results if r != None])
        print(f"[*] Round {server_round} accuracy avg from client results: {100 * accuracies:.2f}%")

        # Weigh TPR of each client by number of examples used
        cls_tprs = np.array([r.metrics["cls_tpr"] * r.num_examples for _, r in results if r != None])
        cls_tprs = cls_tprs[~np.isnan(cls_tprs)]
        # Aggregate and print custom metric
        cls_tpr_aggregated = 100 * sum(cls_tprs) / sum(examples)
        print(f"[*] Round {server_round} TPR weighted avg from client results: {cls_tpr_aggregated:.2f}%")
        cls_tprs = np.mean([r.metrics["cls_tpr"] for _, r in results if r != None])
        print(f"[*] Round {server_round} TPR avg from client results: {100 * cls_tprs:.2f}%")

        # Weigh FPR of each client by number of examples used
        cls_fprs = np.array([(1 - r.metrics["cls_fpr"]) * r.num_examples for _, r in results if r != None])
        cls_fprs = cls_fprs[~np.isnan(cls_fprs)]
        # Aggregate and print custom metric
        cls_fpr_aggregated = 100 * sum(cls_fprs) / sum(examples)
        print(f"[*] Round {server_round} FPR weighted avg from client results: {cls_fpr_aggregated:.2f}%")
        cls_fprs = np.mean([1 - r.metrics["cls_fpr"] for _, r in results if r != None])
        print(f"[*] Round {server_round} FPR avg from client results: {100 * cls_fprs:.2f}%")

        conf_matrices = np.array([deserialize(result.metrics['confusion_matrix']) for _, result in results])
        conf_matrix = conf_matrices.sum(axis=0)
        pprint_cm(conf_matrix, ['Benign', 'Malicious'])

        print("===============================\n   Anomaly detection Results   \n===============================")
        ad_fp = np.sum([r.metrics["ad_fp"] for _, r in results if r != None])
        print(f"[*] Round {server_round} total number of ben. anomalies from client results: {int(ad_fp)}")

        ad_tp = np.sum([r.metrics["ad_tp"] for _, r in results if r != None])
        print(f"[*] Round {server_round} total number of mal. anomalies from client results: {int(ad_tp)}")

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["class_accuracy"] * r.num_examples for _, r in results if r != None]
        examples = [r.num_examples for _, r in results if r != None]

        # Aggregate and print custom metric
        accuracy_aggregated = 100 * sum(accuracies) / sum(examples)
        print(f"[*] Round {server_round} accuracy weighted avg from client results: {accuracy_aggregated:.2f}%")

        accuracies = np.mean([r.metrics["class_accuracy"] for _, r in results if r != None])
        print(f"[*] Round {server_round} accuracy avg from client results: {100 * accuracies:.2f}%")

        # Weigh TPR of each client by number of examples used
        ad_tprs = np.array([r.metrics["ad_tpr"] * r.num_examples for _, r in results if r != None])
        ad_tprs = ad_tprs[~np.isnan(ad_tprs)]
        # Aggregate and print custom metric
        ad_tpr_aggregated = 100 * sum(ad_tprs) / sum(examples)
        print(f"[*] Round {server_round} TPR weighted avg from client results: {ad_tpr_aggregated:.2f}%")
        ad_tprs = np.mean([r.metrics["ad_tpr"] for _, r in results if r != None])
        print(f"[*] Round {server_round} TPR avg from client results: {100 * ad_tprs:.2f}%")

        # Weigh FPR of each client by number of examples used
        ad_fprs = np.array([(1 - r.metrics["ad_fpr"]) * r.num_examples for _, r in results if r != None])
        ad_fprs = ad_fprs[~np.isnan(ad_fprs)]
        # Aggregate and print custom metric
        ad_fpr_aggregated = 100 * sum(ad_fprs) / sum(examples)
        print(f"[*] Round {server_round} FPR weighted avg from client results: {ad_fpr_aggregated:.2f}%")
        ad_fprs = np.mean([1 - r.metrics["cls_fpr"] for _, r in results if r != None])
        print(f"[*] Round {server_round} FPR avg from client results: {100 * ad_fprs:.2f}%")

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(server_round, results, failures)

    def fit_config(self, rnd: int):
        """
        """
        mal_data = load_day_dataset(self.config.vaccine(self.day))
        config = {
            "start_epoch": self.epoch,
            "local_epochs": self.config.local_epochs(rnd),  # 1 if rnd < 2 else 2,
            "threshold": self.threshold,
            "proxy_spheres": serialize(self.proxy_spheres),
            "mal_dataset": serialize(mal_data),
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
            "proxy_spheres": serialize(self.proxy_spheres)
        }
        if rnd > 1:
            config['scaler'] = serialize(self.scaler)

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
            for metric_name, metric_values
            in metrics_transpose.items()
        }
        ignore_set = {'id', 'scaler', 'proxy_spheres', 'tracker', 'confusion_matrix'}

        for metric_name, metric_values in metrics_transpose.items():
            if metric_name in ignore_set:
                continue
            if np.issubdtype(metric_values.dtype, np.integer):
                metrics[metric_name] = metric_values.sum()
            else:
                metrics[metric_name] = np.average(metric_values, weights=client_examples)
        return metrics


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

    model = MultiHeadAutoEncoder(config)
    model.compile()

    if config.load and day > 1 and config.model_file(day - 1).exists():
        model.load_weights(config.model_file(day - 1))
        num_rounds = config.server.num_rounds_other_days
    else:
        num_rounds = config.server.num_rounds_first_day

    print(f"{config.num_fit_clients=}")

    # Create custom strategy that aggregates client metrics
    strategy = AggregateCustomMetricStrategy(
        day=day,
        config=config,
        n_cls_layers=len(model.classifier.weights),
        fraction_fit=config.num_fit_clients / config.num_evaluate_clients,
        fraction_evaluate=1.0,
        min_fit_clients=config.num_fit_clients,
        min_evaluate_clients=config.num_evaluate_clients,
        min_available_clients=config.num_evaluate_clients,
        evaluate_fn=get_eval_fn(model, day, config),
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
        eta=config.server.learning_rate
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

    X_test, y_test = load_data(config.data_dir, day, config.seed)

    # X_test = strategy.scaler.transform(X_test)
    # plot_embedding(X_test, y_test, model)

    # model.set_weights(strategy.best_val_params)
    results.metrics_centralized['best_round'] = strategy.best_round
    results.val_losses_distributed = strategy.val_losses
    results.val_acc_distributed = strategy.val_acc

    model.save_weights(config.model_file(day))
    spheres = model.loss.spheres
    with config.spheres_file(day).open('wb') as f:
        pickle.dump(spheres, f)
    with config.scaler_file(day).open('wb') as f:
        pickle.dump(strategy.scaler, f)
    with config.results_file(day).open('wb') as f:
        pickle.dump(results, f)


def get_eval_fn(model, day: int, experiment_config: Config):
    """Return an evaluation function for server-side evaluation."""

    X_test, y_test = load_data(experiment_config.data_dir, day, experiment_config.seed)

    # The `evaluate` function will be called after every round
    def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar]
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters

        # Read the stored per-feature maximums and minimums
        if server_round > 1:
            with experiment_config.scaler_file(day).open('rb') as f:
                scaler = pickle.load(f)
        else:
            scaler = MinMaxScaler().from_min_max(-10 * np.ones(36), 10 * np.ones(36))

        threshold = config['threshold']

        X_test_ = scaler.transform(X_test)

        # Detect all the samples which are anomalies.
        mse, y_pred, bce = model.eval(X_test_, y_test)

        y_ad_pred = (mse > threshold).astype(float).T
        y_cls_pred = (y_pred > 0.5).astype(float)

        ad_report = classification_report(y_test, y_ad_pred, output_dict=True, labels=[0.0, 1.0],
                                          target_names=['Benign', 'Malicious'])
        supervised_report = classification_report(y_test, y_cls_pred, output_dict=True, labels=[0.0, 1.0],
                                                  target_names=['Benign', 'Malicious'])

        report = {f'ad_{key}': val for key, val in ad_report.items()}
        report.update(
            {f'sup_{key}': val for key, val in supervised_report.items()}
        )
        # Detect all the samples which are anomalies.

        print("Evaluate confusion matrix")
        pprint_cm(confusion_matrix(y_test, y_cls_pred), ['Benign', 'Malicious'])

        return bce + mse.sum(), report

    return evaluate


def load_data(data_dir, day, seed):
    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    # X_train = pd.DataFrame()
    # X_test_ben = pd.DataFrame()
    # X_test_mal = pd.DataFrame()
    # for client_id in range(1, 11):
    #     _, test_temp = load_ben_data(day, client_id, data_dir, drop_labels=False)
    #     mal_temp = load_mal_data(day + 1, data_dir, drop_labels=False)[client_malware_map[client_id]]
    #     # X_train = pd.concat([X_train, train_temp], ignore_index=True)
    #     X_test_ben = pd.concat([X_test_ben, test_temp], ignore_index=True)
    #     X_test_mal = pd.concat([X_test_mal, mal_temp], ignore_index=True)
    # X_test, _, y_test, _ = create_supervised_dataset(X_test_ben, X_test_mal, 0.0)

    ben_train, ben_test = zip(
        *[load_ben_data(day, client, data_dir, drop_labels=False, drop_four_tuple=True) for client in
          range(1, 10)])
    X_ben_test = pd.concat(ben_test, axis=0)
    X_mal_test = pd.concat(load_mal_data(day + 1, data_dir, drop_labels=False, drop_four_tuple=True).values(), axis=0)
    X_test, _, y_test, _ = create_supervised_dataset(X_ben_test, X_mal_test, 0.0, seed)

    return X_test, y_test


if __name__ == "__main__":
    Fire(main)
