from enum import Enum
from functools import reduce
from typing import Optional, List, Iterable, Tuple

import pandas as pd
from pyhocon import ConfigFactory, ConfigTree, HOCONConverter
from pathlib import Path

from pyhocon.config_tree import NonExistentKey


class Setting(Enum):
    FEDERATED = "federated"
    CENTRALIZED = "centralized"
    LOCAL = "local"


class Config(ConfigTree):
    omit_default_key = 'omit_default'

    @staticmethod
    def load(config_path=None, **overrides):
        if overrides.get(Config.omit_default_key):
            config_paths = []
        else:
            config_paths = [Path("default.conf")]

        config_paths += (
            [Path(config_path)] if config_path else []
        )
        config = Config._parse_files(config_paths)
        for key, value in overrides.items():
            config.put(key, value)
        config.__class__ = Config
        config.__init__()
        return config

    def save(self, file):
        file.write(HOCONConverter.to_hocon(self))

    @property
    def data_dir(self) -> Path:
        return Path(self["data_dir"])

    @property
    def malware_dir(self):
        return self.data_dir / "Malware"

    @property
    def experiment_dir(self) -> Path:
        directory = Path(f"{self.output_dir}/experiment_{self.id}")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def model_dir(self) -> Path:
        directory = Path(self.experiment_dir / "model")
        if self.run_id is not None:
            directory = directory / f"{self.run_id:02}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def results_dir(self) -> Path:
        directory = self.experiment_dir / "results"
        if self.run_id is not None:
            directory = directory / f"{self.run_id:02}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def setting(self) -> Setting:
        return Setting(self["setting"])

    def results_file(self, day: int) -> Path:
        return (
                self.results_dir / f"day{day}_{self.seed}_{self.setting.value}_results.pckl"
        )

    def model_file(self, day: int, client_id=None) -> Path:
        if client_id is not None:
            return (
                    self.model_dir
                    / f"day{day}_{self.seed}_{self.setting.value}_{client_id}_model.h5"
            )
        return self.model_dir / f"day{day}_{self.seed}_{self.setting.value}_model.h5"

    def local_model_file(self, day: int, client_id: int) -> Path:
        return (
                self.model_dir
                / f"day{day}_{self.seed}_{self.setting.value}_{client_id}_model.h5"
        )

    def scaler_file(self, day: int) -> Path:
        return self.model_dir / f"day{day}_{self.seed}_{self.setting.value}_scaler.pckl"

    def local_epochs(self, rnd: int) -> int:
        epoch_config = {
            int(key): value for key, value in dict(self.server.local_epochs).items()
        }
        return epoch_config.get(rnd, epoch_config[-1])

    def client_train_malware(self, client_id: int, day: int) -> Iterable[Path]:
        for malware_id, malware_day in self._config_dataset(client_id, day, self.client.client_train_malware):
            if client_id:
                yield self.malware_dir / self.malware_dirs[malware_id] / f"Day{malware_day}"

    def client_test_malware(self, client_id: int, day: int) -> Iterable[Path]:
        for malware_id, malware_day in self._config_dataset(client_id, day, self.client.client_test_malware):
            if client_id:
                yield self.malware_dir / self.malware_dirs[malware_id] / f"Day{malware_day}"

    def client_ben_train(self, client_id: int, day: int) -> Iterable[Path]:
        for client_id, dataset_day in self._config_dataset(client_id, day, self.client.client_train_data):
            if client_id:
                yield self.data_dir / f'Client{client_id}' / f'Day{dataset_day}'

    def client_ben_test(self, client_id: int, day: int) -> Iterable[Path]:
        for client_id, dataset_day in self._config_dataset(client_id, day, self.client.client_test_data):
            if client_id:
                yield self.data_dir / f'Client{client_id}' / f'Day{dataset_day}'

    def _config_dataset(self, client_id: int, day: int, config_item) -> Iterable[Tuple[str, str]]:
        dataset_list = {int(key): value for key, value in config_item.items()}
        if client_id not in dataset_list or dataset_list[client_id][day - 1][0] == "_":
            return

        dataset_list = dataset_list[client_id][day - 1]
        if not isinstance(dataset_list, list):
            dataset_list = [dataset_list]
        for dataset in dataset_list:
            if "__" in dataset:
                continue
            if "_" in dataset:
                dataset_id, day = dataset.split("_")
            else:
                dataset_id, day = dataset, day
            yield dataset_id, day


    def vaccine(self, day: int) -> List[Path]:
        if "vaccine_malware" not in self.server:
            return None
        return self.get_malware_list(self.server.vaccine_malware, day)
    
    def holdout(self, day: int) -> List[Path]:
        if "holdout_malware" not in self.server:
            return None
        return self.get_malware_list(self.server.holdout_malware, day)
    
    def get_malware_list(self, config_key: list, day: int) -> List[Path]:
        malware_names = config_key[day - 1]
        if isinstance(malware_names, str):
            malware_names = [malware_names]
            
        malware_paths = []
        for malware_name in malware_names:
            if malware_name == "_":
                return None
            malware_id, malware_day = malware_name.split("_")
            malware_path = self.malware_dir / self.malware_dirs[malware_id] / f"Day{malware_day}"
            malware_paths.append(malware_path)
        return malware_paths

    @staticmethod
    def _parse_files(files):
        configs = []
        for file in files:
            configs.append(ConfigFactory.parse_file(file))

        config = configs[0]
        for other_config in configs[1:]:
            ConfigTree.merge_configs(config, other_config)
        return config

    def __getattr__(self, item):
        val = self.get(item, NonExistentKey)
        if val is NonExistentKey:
            return super(Config, self).__getattr__(item)
        return val
