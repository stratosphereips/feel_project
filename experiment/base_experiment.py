from contextlib import contextmanager
from multiprocessing import Process
from pathlib import Path

from common.config import Config, Setting
from time import sleep
import random
import shutil


class BaseExperiment:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = Config.load(config_path)

    def run(self):
        if self.config.experiment_dir.exists():
            shutil.rmtree(self.config.experiment_dir, ignore_errors=True)

        for run in range(self.config.num_runs):
            with self.config_with_random_seed(run) as config_path:
                for day in range(1, self.config.days + 1):
                    self.run_day(day, config_path)

                for day in range(1, self.config.days + 1):
                    self.run_centralized(day, config_path)

                if self.config.evaluate_local_setting:
                    for day in range(1, self.config.days + 1):
                        self.run_day(day, config_path, local=True)

        done_file: Path = self.config.experiment_dir / "done"
        done_file.write_text("DONE")

    def rerun_local(self):
        assert self.config.experiment_dir.exists()

        for run in range(self.config.num_runs):
            exp_dir = self.config.experiment_dir / "results" / f"{run:02}"
            _, seed, *_ = list(exp_dir.iterdir())[0].name.split("_")
            seed = int(seed)
            with self.config_with_random_seed(run, seed) as config_path:
                if self.config.evaluate_local_setting:
                    for day in range(1, self.config.days + 1):
                        self.run_day(day, config_path, local=True)

        done_file: Path = self.config.experiment_dir / "done_local"
        done_file.write_text("DONE")

    def rerun_centralized(self):
        assert self.config.experiment_dir.exists()

        for run in range(self.config.num_runs):
            exp_dir = self.config.experiment_dir / "results" / f"{run:02}"
            _, seed, *_ = list(exp_dir.iterdir())[0].name.split("_")
            seed = int(seed)
            with self.config_with_random_seed(run, seed) as config_path:
                for day in range(1, self.config.days + 1):
                    self.run_centralized(day, config_path)

        done_file: Path = self.config.experiment_dir / "done_local"
        done_file.write_text("DONE")

    def rerun_federated(self):
        assert self.config.experiment_dir.exists()

        for run in range(self.config.num_runs):
            exp_dir = self.config.experiment_dir / "results" / f"{run:02}"
            _, seed, *_ = list(exp_dir.iterdir())[0].name.split("_")
            seed = int(seed)
            with self.config_with_random_seed(run, seed) as config_path:
                if self.config.evaluate_local_setting:
                    for day in range(1, self.config.days + 1):
                        self.run_day(day, config_path)

        done_file: Path = self.config.experiment_dir / "done_federated"
        done_file.write_text("DONE")

    def run_day(self, day: int, config_path: Path, local=False):
        if local:
            kwargs = {
                "setting": Setting.LOCAL.value,
                "model.proximal": False,
                "model.optimizer": "Adam",
            }
        else:
            kwargs = {"setting": Setting.FEDERATED.value}

        server_process = Process(
            target=self.server_target, args=(day, config_path), kwargs=kwargs
        )

        client_processes = [
            Process(
                target=self.client_target,
                args=(client_id, day, config_path),
                kwargs=kwargs,
            )
            for client_id in range(1, self.config.num_evaluate_clients + 1)
        ]

        print(f"Starting server for day {day}")
        server_process.start()

        sleep(5)

        for i, client_process in enumerate(client_processes):
            print(f"Starting Client {i}")
            client_process.start()

        for client_process in client_processes:
            client_process.join()
        print("Clients terminated")

        server_process.join()
        print("Server terminated")

    def run_centralized(self, day: int, config_path: Path):
        local_process = Process(
            target=self.centralized_target,
            args=(day, config_path),
            kwargs={
                "setting": Setting.CENTRALIZED.value,
                "model.optimizer": "Adam",
                "model.proximal": False,
            },
        )
        print(f"Starting centralized for day {day}")
        local_process.start()
        local_process.join()
        print(f"Centralized for day {day} terminated")

    @contextmanager
    def config_with_random_seed(self, run_num, seed=None):
        seed = random.randint(0, int(1e9)) if seed is None else seed
        new_config_path = (
            self.config_path.parent
            / f"{self.config_path.name}_{seed:09}{self.config_path.suffix}"
        )

        new_config = self.config.copy()
        new_config["seed"] = seed
        new_config["run_id"] = run_num
        new_config["port"] = 8000 + random.randint(1, 999)
        with new_config_path.open("w") as f:
            new_config.save(f)

        try:
            yield new_config_path
        finally:
            new_config_path.unlink()

    @property
    def server_target(self):
        raise NotImplemented

    @property
    def client_target(self):
        raise NotImplemented

    def centralized_target(self):
        raise NotImplemented
