import pandas as pd
from pathlib import Path
from fire import Fire


def main(in_dir, out_dir):
    """
    Combine all csv files with features to one
    @param in_dir: Path to a folder where all csv files are.
    @param out_dir: Where to store the final output
    """
    df = combine_files(Path(in_dir))
    out_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(Path(out_dir / "comb_features.csv"), index=False)


def combine_files(folder_name: Path):
    df = pd.DataFrame()
    total_lines = 0
    for folder in sorted(folder_name.iterdir()):
        try:
            if (folder / 'features.csv').exists():
                df_temp = pd.read_csv(folder / 'features.csv')
                total_lines += len(df_temp)
                df = pd.concat([df, df_temp], ignore_index=True)
        except NotADirectoryError:
            pass

    assert total_lines == len(df)
    return df


if __name__ == '__main__':
    Fire(main)
