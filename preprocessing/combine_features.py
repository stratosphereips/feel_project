import pandas as pd
from pathlib import Path
from fire import Fire


def main(in_dir, out_dir, ip_map=None, day=0):
    """
    Combine all csv files with features to one
    @param in_dir: Path to a folder where all csv files are.
    @param out_dir: Where to store the final output
    """
    if ip_map is None:
        ip_map = {}

    df = combine_files(Path(in_dir), ip_map, day)
    out_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(Path(out_dir / "comb_features.csv"), index=False)


def combine_files(folder_name: Path, ip_map, day):
    df = pd.DataFrame()
    total_lines = 0
    for folder in sorted(folder_name.iterdir()):
        try:
            if (folder / "features.csv").exists():
                df_temp = pd.read_csv(folder / "features.csv")
                hour = int(folder.name) % (24 * 3600) // 3600

                df_temp["id.orig_h"] = df_temp["id.orig_h"].transform(
                    ip_map.__getitem__
                )
                df_temp["day"] = day
                df_temp["hour"] = hour

                total_lines += len(df_temp)
                df = pd.concat([df, df_temp], ignore_index=True)
        except NotADirectoryError:
            pass

    assert total_lines == len(df)
    return df


if __name__ == "__main__":
    Fire(main)
