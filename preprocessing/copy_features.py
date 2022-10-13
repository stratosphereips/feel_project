from pathlib import Path
from fire import Fire
from shutil import copy


def main(raw_dir, processed_dir, features_filename='comb_features.csv'):
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    for i, client_dir in enumerate(sorted(raw_dir.iterdir())):
        for day_dir in client_dir.iterdir():
            if not day_dir.is_dir():
                continue

            target_dir = processed_dir / f'Client{i}' / day_dir.name
            target_dir.mkdir(exist_ok=True, parents=True)
            copy(day_dir / features_filename, target_dir / features_filename)


if __name__ == '__main__':
    Fire(main)