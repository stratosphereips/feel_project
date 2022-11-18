from pyhocon import ConfigFactory, ConfigTree
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

    @property
    def experiment_dir(self) -> Path:
        directory = Path(f'experiment_{self.id}_{self.seed}')
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def model_dir(self) -> Path:
        directory = Path(self.experiment_dir / 'model')
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def results_dir(self) -> Path:
        directory = self.experiment_dir / 'results'
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def results_file(self, day: int) -> Path:
        return self.results_dir / f'day{day}_seed-{self.seed}_results.pckl'

    def model_file(self, day: int) -> Path:
        return self.model_dir / f'day{day}_{self.seed}_model.h5'

    def scaler_file(self, day: int) -> Path:
        return self.model_dir / f'day{day}_{self.seed}_scaler.pckl'

    def local_epochs(self, rnd: int) -> int:
        epoch_config = {int(key): value for key, value in dict(self.server.local_epochs).items()}
        return epoch_config.get(rnd, epoch_config[-1])

    @staticmethod
    def _parse_files(files):
        contents = []
        for file in files:
            with file.open('r') as f:
                contents.append(f.read())
        return ConfigFactory.parse_string("\n".join(contents))

    def __getattr__(self, item):
        val = self.get(item, NonExistentKey)
        if val is NonExistentKey:
            return super(Config, self).__getattr__(item)
        return val