import fire
from fire import Fire
import pandas as pd
from zat.log_to_dataframe import LogToDataFrame
from pathlib import Path


class FeatureExtractor:
    def __init__(
            self,
            zeek_logs_dir,
            output_file='features.csv',
            conn_file='conn.log.labeled',
            ssl_file='ssl.log.labeled',
            x509_file='x509.log'
    ):
        self._zeek_logs_dir = Path(zeek_logs_dir).resolve()
        self._output_file = Path(output_file).resolve()
        self._conn_path = self._zeek_logs_dir / conn_file
        self._ssl_path = self._zeek_logs_dir / ssl_file
        self._x509_path = self._zeek_logs_dir / x509_file
        self._files_exist()

        conn_df = self._load_zeek(self._conn_path)
        ssl_df = self._load_zeek(self._ssl_path)
        self.flows_df = conn_df.reset_index().set_index('uid').join(ssl_df.set_index('uid'), rsuffix='_ssl_flows')
        self.flows_df.reset_index().set_index('ts')
        self.flows_df = self.flows_df[[col for col in self.flows_df.columns if '_ssl_flows' not in col]]

        self._cert_df = self._load_zeek(self._x509_path)

    def _extract(self):
        pass

    def _files_exist(self):
        for file in [self._conn_path, self._ssl_path, self._x509_path]:
            if not file.exists():
                raise FileNotFoundError(f"File {file} does not exist")

    def _load_zeek(self, file) -> pd.DataFrame:
        return LogToDataFrame().create_dataframe(file)



def main(zeek_logs_dir,
         output_file='features.csv',
         conn_file='conn.log.labeled',
         ssl_file='ssl.log.labeled',
         x509_file='x509.log'):
    FeatureExtractor(zeek_logs_dir, output_file, conn_file, ssl_file, x509_file)


if __name__ == '__main__':
    fire.Fire(main)
