from functools import reduce
from typing import List, Tuple, Optional, Union, Dict

import flwr as fl
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, NDArrays, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate


class CustomFedAdam(fl.server.strategy.FedAdam):
    def __init__(self, n_cls_layers=0, **kwargs):
        self.n_cls_layers = n_cls_layers
        super().__init__(**kwargs)

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException],
    ) -> Optional[fl.common.Parameters]:
        fedavg_parameters_aggregated, metrics_aggregated = self._aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )
        if fedavg_parameters_aggregated is None:
            return None, {}

        fedavg_weights_aggregate = parameters_to_ndarrays(fedavg_parameters_aggregated)

        # Adam
        delta_t: NDArrays = [
            x - y for x, y in zip(fedavg_weights_aggregate, self.current_weights)
        ]

        # m_t
        if not self.m_t:
            self.m_t = [np.zeros_like(x) for x in delta_t]
        self.m_t = [
            np.multiply(self.beta_1, x) + (1 - self.beta_1) * y
            for x, y in zip(self.m_t, delta_t)
        ]

        # v_t
        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]
        self.v_t = [
            self.beta_2 * x + (1 - self.beta_2) * np.multiply(y, y)
            for x, y in zip(self.v_t, delta_t)
        ]

        new_weights = [
            x + self.eta * y / (np.sqrt(z) + self.tau)
            for x, y, z in zip(self.current_weights, self.m_t, self.v_t)
        ]

        self.current_weights = new_weights

        return ndarrays_to_parameters(self.current_weights), metrics_aggregated

    def _aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            # (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, fit_res.metrics['num_mal_examples_train'])
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(self._aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated

    @staticmethod
    def _aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
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
    #
    # def _aggregate(self, results: List[Tuple[NDArrays, int, int]]) -> NDArrays:
    #     """Compute weighted average."""
    #     # Calculate the total number of examples used during training
    #     # num_examples_total = sum([num_examples for _, num_examples, _ in results])
    #     # num_mal_examples_total = sum([num_examples for _, _, num_examples in results])
    #     n_layers = len(results[0][0])
    #
    #     per_layer_examples = []
    #     for layers, num_examples, num_mal_examples in results:
    #         layer_examples = []
    #         for layer_num, layer in enumerate(layers):
    #             if not layer.any():
    #                 layer_examples.append(0)
    #             elif layer_num >= n_layers - self.n_cls_layers:
    #                 layer_examples.append(num_mal_examples)
    #             else:
    #                 layer_examples.append(num_examples)
    #         per_layer_examples.append(layer_examples)
    #
    #     layer_examples = [sum(examples) for examples in zip(*per_layer_examples)]
    #     # Create a list of weights, each multiplied by the related number of examples
    #     weighted_weights = [
    #         [layer * num_examples for layer, num_examples in zip(weights, client_examples)]
    #         for (weights, _, _), client_examples in zip(results, per_layer_examples)
    #     ]
    #
    #     # Compute average weights of each layer
    #     weights_prime: NDArrays = [
    #         reduce(np.add, layer_updates) / (num_examples if num_examples > 0 else 1.0)
    #         for num_examples, *layer_updates in zip(layer_examples, *weighted_weights)
    #     ]
    #     return weights_prime

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, self.on_evaluate_config_fn(0))
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics