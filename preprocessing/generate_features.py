from pathlib import Path
from fire import Fire
from tempfile import TemporaryDirectory
from create_windows import _split_file
import combine_features
import subprocess
from shutil import copy
from tqdm import tqdm


class FeaturesGenerator:
    def __init__(self, raw_dir, processed_dir, time_window=3600):
        """
        Script to generate ssl-based features from the raw zeek logs.

        @param raw_dir: Path to the input directory with the raw data
        @param processed_dir: Path where the processed features will be stored
        @param time_window: Size of the time window in seconds - default 1 hour
        """
        self._raw_dir = Path(raw_dir)
        self._processed_dir = Path(processed_dir)
        self._time_window = time_window

        self._normal_dir = self._raw_dir / 'Normal'
        self._malware_dir = self._raw_dir / 'Malware'

        self._script_dir = Path(__file__).parent
        self._extractor_path = self._script_dir / '..' / 'feature_extractor' / 'feature_extractor.py'

    def generate_client_features(self):
        for i, client_dir in tqdm(list(enumerate(sorted(self._normal_dir.iterdir()), start=1)), "Client"):
            for day_dir in tqdm(list(client_dir.iterdir()), "Day"):
                if not day_dir.is_dir():
                    continue

                target_dir = self._processed_dir / f'Client{i}' / day_dir.name
                target_dir.mkdir(exist_ok=True, parents=True)
                self._generate_features(day_dir, target_dir)

    def generate_malware_features(self):
        for malware_dir in tqdm(list(sorted(self._malware_dir.iterdir())), "Malware"):
            for day_dir in tqdm(list(malware_dir.iterdir()), "Day"):
                if not day_dir.is_dir() and day_dir.name != 'zeek':
                    continue

                target_dir = self._processed_dir / 'Malware' / malware_dir.name / day_dir.name
                target_dir.mkdir(exist_ok=True, parents=True)
                self._generate_features(day_dir, target_dir)

    def _generate_features(self, logs_dir: Path, output_dir: Path):
        with TemporaryDirectory() as tmpdir:
            temp_dir_path = Path(tmpdir)
            for file_type in ['ssl', 'conn']:
                log_file = logs_dir / f'{file_type}.log.labeled'
                if log_file.stat().st_size <= 700:
                    # this is kinda a magic number, but it's approximately the number of bytes in the header -
                    # if there is only the header, just skip it.
                    return
                _split_file(logs_dir / f'{file_type}.log.labeled', self._time_window, temp_dir_path)

            for dir in tqdm(list(temp_dir_path.iterdir()), "Time bucket"):
                copy(logs_dir / 'x509.log', dir)
                subprocess.call(['python', str(self._extractor_path), '-v', '0', '-z', str(dir)])

            combine_features.main(temp_dir_path, output_dir)


if __name__ == '__main__':
    Fire(FeaturesGenerator)