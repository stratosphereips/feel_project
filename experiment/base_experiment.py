from contextlib import contextmanager
from multiprocessing import Process
from pathlib import Path

from common.config import Config
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
                for day in range(1, self.config.days+1):
                    self.run_day(day, config_path)

    def run_day(self, day: int, config_path: Path):
        server_process = Process(target=self.server_target, args=(day, config_path))

        client_processes = [Process(target=self.client_target, args=(client_id, day, config_path))
                            for client_id in range(1, self.config.num_evaluate_clients + 1)]

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

    @contextmanager
    def config_with_random_seed(self, run_num):
        seed = random.randint(0, int(1e9))
        new_config_path = self.config_path.parent / f'{self.config_path.name}_{seed:09}{self.config_path.suffix}'

        new_config = self.config.copy()
        new_config['seed'] = seed
        new_config['run_id'] = run_num
        new_config['port'] = 8000 + random.randint(1, 999)
        with new_config_path.open('w') as f:
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










