from pathlib import Path

from experiment.base_experiment import BaseExperiment
import server
import client
import central_scenario
import fire


class AnomalyDetectionExperiment(BaseExperiment):
    @property
    def server_target(self):
        return server.main

    @property
    def client_target(self):
        return client.main

    @property
    def centralized_target(self):
        return central_scenario.main


def main(config_path: str):
    exp = AnomalyDetectionExperiment(Path(config_path))
    exp.run()


if __name__ == '__main__':
    fire.Fire(main)