from pathlib import Path

from experiment.base_experiment import BaseExperiment
import server
import client
import fire


class SupervisedExperiment(BaseExperiment):
    @property
    def server_target(self):
        return server.main

    @property
    def client_target(self):
        return client.main

def main(config_path: str):
    exp = SupervisedExperiment(Path(config_path))
    exp.run()

if __name__ == '__main__':
    fire.Fire(main)