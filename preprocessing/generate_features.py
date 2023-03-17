from collections import defaultdict
from pathlib import Path

import pandas as pd
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

        self._normal_dir = self._raw_dir / "Normal"
        self._malware_dir = self._raw_dir / "Malware"

        self._script_dir = Path(__file__).parent
        self._extractor_path = (
            self._script_dir / ".." / "feature_extractor" / "feature_extractor.py"
        )

        self.ip_map = keydefaultdict(IpObfuscationDefaultFactory.new)

    def generate_features(self):
        self.generate_client_features()
        self.generate_malware_features()
        self._save_ip_map(Path("ip_obfuscation_mapping.csv"))

    def generate_client_features(self):
        client_pbar = tqdm(list(enumerate(sorted(self._normal_dir.iterdir()), start=1)))
        for i, client_dir in client_pbar:
            client_pbar.set_description(client_dir.name)
            day_pbar = tqdm(sorted(client_dir.iterdir()))
            for day_dir in day_pbar:
                day_pbar.set_description(day_dir.name)
                if not day_dir.is_dir():
                    continue

                target_dir = self._processed_dir / f"Client{i}" / day_dir.name
                target_dir.mkdir(exist_ok=True, parents=True)
                self._generate_features(day_dir, target_dir)

    def generate_malware_features(self):
        mal_pbar = tqdm(list(sorted(self._malware_dir.iterdir())))
        for malware_dir in mal_pbar:
            mal_pbar.set_description(malware_dir.name)
            day_pbar = tqdm(sorted(malware_dir.iterdir()))
            for day_dir in day_pbar:
                day_pbar.set_description(day_dir.name)
                if not day_dir.is_dir() and day_dir.name != "zeek":
                    continue

                target_dir = (
                    self._processed_dir / "Malware" / malware_dir.name / day_dir.name
                )
                target_dir.mkdir(exist_ok=True, parents=True)
                self._generate_features(day_dir, target_dir)

    def _generate_features(self, logs_dir: Path, output_dir: Path):
        with TemporaryDirectory() as tmpdir:
            temp_dir_path = Path(tmpdir)
            for file_type in ["ssl", "conn"]:
                log_file = logs_dir / f"{file_type}.log.labeled"
                if log_file.stat().st_size <= 700:
                    # this is kinda a magic number, but it's approximately the number of bytes in the header -
                    # if there is only the header, just skip it.
                    return
                filter_invalid_malware = "CTU-Malware-Capture-Botnet-346-1" in str(
                    output_dir
                )
                _split_file(
                    logs_dir / f"{file_type}.log.labeled",
                    self._time_window,
                    temp_dir_path,
                    filter_invalid_malware,
                )

            for directory in tqdm(list(temp_dir_path.iterdir()), "Time bucket"):
                copy(logs_dir / "x509.log", directory)
                subprocess.call(
                    [
                        "python",
                        str(self._extractor_path),
                        "-v",
                        "0",
                        "-z",
                        str(directory),
                    ]
                )

            combine_features.main(temp_dir_path, output_dir, self.ip_map)

    def _save_ip_map(self, file):
        df = pd.DataFrame(self.ip_map.items(), columns=["orig_ip", "obfuscated_ip"])
        df.to_csv(file, index=False)


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class IpObfuscationDefaultFactory:
    count = 0

    @staticmethod
    def new(ip: str):
        *prefix, last_oct = ip.split(".")
        obfuscated_ip = ".".join(prefix + [f"x{IpObfuscationDefaultFactory.count}"])
        IpObfuscationDefaultFactory.count += 1
        return obfuscated_ip


if __name__ == "__main__":
    Fire(FeaturesGenerator)
