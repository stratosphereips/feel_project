import argparse
import os
import pandas as pd

def read_args():

    # Parse the parameters
    parser = argparse.ArgumentParser(description="Combine all csv files with features to one")
    parser.add_argument('-i', '--in_dir', help='Path to a folder where all csv files are.', action='store',
                        required=True)
    parser.add_argument('-o', '--out_dir', required=True, help="Where to store the final output", action='store')
    args = parser.parse_args()

    return args

def combine_files(folder_name):
    df = pd.DataFrame()
    total_lines = 0
    folders = os.listdir(folder_name)
    for folder in folders:
        try:
            if 'features.csv' in os.listdir(os.path.join(folder_name, folder)):
                df_temp = pd.read_csv(os.path.join(folder_name, folder, 'features.csv'))
                total_lines += len(df_temp)
                df = pd.concat([df, df_temp], ignore_index=True)
        except NotADirectoryError:
            pass

    assert total_lines == len(df)
    return df

if __name__ == '__main__':
    args = read_args()

    df = combine_files(args.in_dir)
    df.to_csv(os.path.join(args.out_dir, "comb_features.csv"), index=False)


