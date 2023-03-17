from pathlib import Path
from fire import Fire
import pandas as pd
from zat.log_to_dataframe import LogToDataFrame
from typing import List


def main(time_window: int, output_dir: str, *args: List[str]):
    """
    Splits zeek log files based on time-windows
    time_window : int
        Duration of time window in seconds
    output_dir : str
        directory into which the new generated log files will be writen
    *args : List[str]
        list of paths to log files
    """
    for file_path in args:
        _split_file(Path(file_path), time_window, Path(output_dir))


def _split_file(
    log_file_path: Path, freq: int, out_dir: Path, filter_out_invalid_malware=False
):
    header = _get_header(log_file_path)
    if not header:
        print(f"File is empty")
        return

    df = LogToDataFrame().create_dataframe(log_file_path)
    if filter_out_invalid_malware:
        df = _filter_out_invalid_malware(df)
    if "duration" in df.columns:
        df["duration"] = df["duration"].transform(lambda x: x.total_seconds())

    for label, bucket_df in list(df.groupby(pd.Grouper(freq=f"{freq}s", label="left"))):
        if bucket_df.size == 0:
            continue
        ts = int(label.timestamp())
        bucket_out_dir = out_dir / f"{ts:010}"
        bucket_out_dir.mkdir(parents=True, exist_ok=True)
        filename = log_file_path.name.split(".")
        filename = ".".join(filename[:1] + [f"{ts}-{ts + freq - 1}"] + filename[1:])

        bucket_df.index = bucket_df.index.map(lambda x: x.timestamp())

        out_file = bucket_out_dir / filename
        with out_file.open("w") as f:
            f.writelines(header)
        bucket_df.to_csv(out_file, mode="a", header=False, sep="\t")


def _get_header(file_path: Path):
    header = []
    with file_path.open("r") as f:
        for line in f:
            if line[0] != "#":
                break
            header.append(line)
    return header


def _filter_out_invalid_malware(df):
    invalid_malware_mask = (df["id.resp_p"] == 443) & (
        df["detailedlabel"] == "CC-with-MITM-from-analysts"
    )
    return df[~invalid_malware_mask]


if __name__ == "__main__":
    Fire(main)
