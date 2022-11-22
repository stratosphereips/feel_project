from functools import reduce
from typing import Optional

from pyhocon import ConfigFactory, ConfigTree, HOCONConverter
from pathlib import Path

from pyhocon.config_tree import NonExistentKey


class Config(ConfigTree):
    @staticmethod
    def load(config_path=None, **overrides):
        config_paths = [Path('default.conf')] + ([Path(config_path)] if config_path else [])
        config = Config._parse_files(config_paths)
        for key, value in overrides.items():
            config['key'] = value
        config.__class__ = Config
        config.__init__()
        return config

    def save(self, file):
        file.write(HOCONConverter.to_hocon(self))

    @property
    def data_dir(self) -> Path:
        return Path(self['data_dir'])

    @property
    def malware_dir(self):
        return self.data_dir / 'Processed' / 'Malware'

    @property
    def experiment_dir(self) -> Path:
        directory = Path(f'experiment_{self.id}')
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def model_dir(self) -> Path:
        directory = Path(self.experiment_dir / 'model')
        if self.run_id is not None:
            directory = directory / f'{self.run_id:02}'
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def results_dir(self) -> Path:
        directory = self.experiment_dir / 'results'
        if self.run_id is not None:
            directory = directory / f'{self.run_id:02}'
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def results_file(self, day: int) -> Path:
        return self.results_dir / f'day{day}_seed-{self.seed}_results.pckl'

    def model_file(self, day: int) -> Path:
        return self.model_dir / f'day{day}_{self.seed}_model.h5'

    def scaler_file(self, day: int) -> Path:
        return self.model_dir / f'day{day}_{self.seed}_scaler.pckl'

    def spheres_file(self, day: int) -> Path:
        return self.model_dir / f'day{day}_{self.seed}_spheres.pckl'

    def local_epochs(self, rnd: int) -> int:
        epoch_config = {int(key): value for key, value in dict(self.server.local_epochs).items()}
        return epoch_config.get(rnd, epoch_config[-1])

    def client_malware(self, client_id: int, day: int) -> Optional[Path]:
        malware = {int(key): value for key, value in self.client.client_malware.items()}
        if client_id not in malware:
            return None

        malware_id = malware[client_id][day-1]
        return self.malware_dir / self.malware_dirs[malware_id] / f'Day{day}'

    def vaccine(self, day: int) -> Path:
        malware_id, malware_day = self.server.vaccine_malware[day-1].split('_')
        # print(f"====\n\n\n Reading vaccine from {self.malware_dir / self.malware_dirs[malware_id] / f'Day{malware_day}'}\n\n\n")
        return self.malware_dir / self.malware_dirs[malware_id] / f'Day{malware_day}'

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