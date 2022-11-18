import inspect
import os
import types

from common.CustomFedAdam import CustomFedAdam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings

warnings.filterwarnings('ignore')

from functools import reduce

from flwr.common import FitRes, Parameters, Scalar, NDArrays, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

from typing import Dict, Optional, Tuple, List, Union

import flwr as fl
import numpy as np
from pathlib import Path
from fire import Fire

from common.utils import serialize_array, deserialize_string, client_malware_map, scale_data, MinMaxScaler, serialize, \
    deserialize, pprint_cm, plot_embedding
from common.models import get_classification_model, MultiHeadAutoEncoder, MetricsTracker
from common.data_loading import load_mal_data, load_ben_data, create_supervised_dataset
import pickle
import pandas as pd
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.metrics import Metric


class AggregateCustomMetricStrategy(CustomFedAdam):
    def __init__(self, model, **kwds):
        self.model = model
        kwds['on_fit_config_fn'] = self.fit_config
        kwds['on_evaluate_config_fn'] = self.evaluate_config
        super().__init__(**kwds)
        self.prev_median_loss = np.inf
        self.proxy_spheres = {}
        self.scaler = None
        self.epoch = 0
        self.threshold = 100

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException],
    ) -> Optional[fl.common.Parameters]:
        results = [result for result in results if result[1].num_examples]
        results = sorted(results, key=lambda result: result[1].metrics['id'])
        if server_round == 1:
            self.scaler = sum((MinMaxScaler.load(r.metrics['scaler']) for _, r in results), start=MinMaxScaler())
            with open('round-1-scaler.pckl', 'wb') as fb:
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
        self.model.set_spheres(self.proxy_spheres)
        print(f'Aggregating {len(results)} results')
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        if aggregated_weights is not None and server_round == 9:
            # Save aggregated_weights on the final round
            print(f"[*] Saving round {server_round} aggregated_weights...")
            np.savez(f"model-weights.npz", *aggregated_weights)

        self.epoch += 1

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
        """Return training configuration dict for each round.
        Keep batch size fixed at 32, perform two rounds of training with one
        local epoch, increase to two local epochs afterwards.
        """
        mal_data = load_mal_data(5, Path('../data'))["CTU-Malware-Capture-Botnet-67-1"]
        config = {
            "start_epoch": self.epoch,
            "batch_size": 32,
            "local_epochs": 1,  # 1 if rnd < 2 else 2,
            "neg_dataset": serialize(mal_data),
            "proxy_spheres": serialize(self.proxy_spheres)
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
            with open('round-1-scaler.pckl', 'rb') as f:
                scaler = pickle.load(f)
                config['scaler'] = scaler.dump()

        return config


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

    model = MultiHeadAutoEncoder()
    model.compile()
    model.predict(np.zeros((16, 36)))
    num_rounds = 25
    print(f"{num_clients=}")

    # Create custom strategy that aggregates client metrics
    strategy = AggregateCustomMetricStrategy(
        model=model,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=get_eval_fn(model, day, Path(data_dir), seed),
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
        eta=1e-2
    )

    # Start Flower server (SSL-enabled) for n rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8000",
        config=fl.server.ServerConfig(num_rounds=num_rounds, round_timeout=90.0),
        strategy=strategy,
        # certificates=(
        #    Path(".cache/certificates/ca.crt").read_bytes(),
        #    Path(".cache/certificates/server.pem").read_bytes(),
        #    Path(".cache/certificates/server.key").read_bytes(),
        # ),
    )

    X_test, y_test = load_data(data_dir, day, seed)
    with open(f'round-1-scaler.pckl', 'rb') as f:
        scaler = pickle.load(f)
    X_test = strategy.scaler.transform(X_test)
    plot_embedding(X_test, y_test, model)

    model.save_weights(f'day{day}_{seed}_model.h5')
    spheres = model.loss.spheres
    with open(f'day{day}_{seed}_spheres.h5', 'wb') as f:
        pickle.dump(spheres, f)
    with open(f'round-1-scaler.pckl', 'rb') as f:
        scaler = pickle.load(f)


def get_eval_fn(model, day, data_dir, seed):
    """Return an evaluation function for server-side evaluation."""

    X_test, y_test = load_data(data_dir, day, seed)

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

        threshold = config['threshold']

        X_test_ = scaler.transform(X_test)

        # Detect all the samples which are anomalies.
        mse, y_pred, bce = model.eval(X_test_, y_test)

        y_ad_pred = (mse > threshold).astype(float).T

        ad_report = classification_report(y_test, y_ad_pred, output_dict=True, label=[0.0, 1.0],
                                          target_names=['Benign', 'Malicious'])
        supervised_report = classification_report(y_test, y_pred, output_dict=True, label=[0.0, 1.0],
                                          target_names=['Benign', 'Malicious'])

        report = {f'ad_{key}': val for key, val in ad_report.items()}.update(
            {f'sup_{key}': val for key, val in supervised_report.items()}
        )
        # Detect all the samples which are anomalies.

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


def mergeFunctionMetadata(f, g):
    """
    Overwrite C{g}'s name and docstring with values from C{f}.  Update
    C{g}'s instance dictionary with C{f}'s.
    To use this function safely you must use the return value. In Python 2.3,
    L{mergeFunctionMetadata} will create a new function. In later versions of
    Python, C{g} will be mutated and returned.
    @return: A function that has C{g}'s behavior and metadata merged from
        C{f}.
    """
    try:
        g.__name__ = f.__name__
    except TypeError:
        try:
            merged = types.FunctionType(
                g.func_code, g.func_globals,
                f.__name__, inspect.getargspec(g)[-1],
                g.func_closure)
        except TypeError:
            pass
    else:
        merged = g
    try:
        merged.__doc__ = f.__doc__
    except (TypeError, AttributeError):
        pass
    try:
        merged.__dict__.update(g.__dict__)
        merged.__dict__.update(f.__dict__)
    except (TypeError, AttributeError):
        pass
    merged.__module__ = f.__module__
    return merged


def custom_aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    per_layer_examples = [
        [num_examples if layer.any() else 0 for layer in layers]
        for layers, num_examples in results
    ]
    layer_examples = [sum(examples) for examples in zip(*per_layer_examples)]
    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_examples, *layer_updates in zip(layer_examples, *weighted_weights)
    ]
    return weights_prime


mergeFunctionMetadata(custom_aggregate, fl.server.strategy.aggregate.aggregate)

if __name__ == "__main__":
    Fire(main)
