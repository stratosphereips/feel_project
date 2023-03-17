import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path

from common.utils import get_threshold
from common.models import get_ad_model
from common.data_loading import load_mal_data, load_ben_data

data_dir = Path("../data")

# tf.config.run_functions_eagerly(True)


def create_dataset(day, num_clients=10):
    clients = np.random.choice(range(1, 11), num_clients, replace=False)
    print(clients)
    data = dict()
    # for j in range(1, 6):
    data = pd.DataFrame()
    for i in clients:
        df_temp, _ = load_ben_data(day, i, data_dir)
        data = pd.concat([data, df_temp], ignore_index=True)

    mal_data = load_mal_data(1, data_dir)

    return data, mal_data


def metrics(fp, tp, num_ben, num_malware):
    # fp = anomalies_ben
    # tp = anomalies_mal
    tn = num_ben - fp
    # fn = num_malware - tp

    accuracy = (tp + tn) / (num_ben + num_malware)
    tpr = tp / num_malware
    fpr = fp / num_ben

    return accuracy, tpr, fpr


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Centralized training scenario 1a")
    parser.add_argument(
        "--seed", type=int, required=False, default=8181, help="Random state seed."
    )
    parser.add_argument(
        "--num_clients",
        "-n",
        type=int,
        choices=(range(1, 11)),
        required=False,
        default=10,
    )

    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    model = get_ad_model()

    for day in range(1, 5):
        data, mal_data = create_dataset(day, args.num_clients)
        mal_folders = list(mal_data.keys())

        if day == 1:
            EPOCHS = 8
        else:
            EPOCHS = 3

        BATCH_SIZE = 64

        scaler = MinMaxScaler()
        X = scaler.fit_transform(data)
        X_test, _ = create_dataset(day + 1, 10)
        X_test = scaler.transform(X_test)

        X_train, X_val = train_test_split(X, test_size=0.2, random_state=args.seed)

        model.fit(
            X_train,
            X_train,
            shuffle=True,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, X_val),
        )

        rec_ben = model.predict(X_test)
        mse_ben = np.mean(np.power(X_test - rec_ben, 2), axis=1)

        rec_mal = dict()
        mse_mal = dict()
        num_malware = 0
        for folder in mal_folders:
            X_test_mal = scaler.transform(mal_data[folder])
            num_malware += X_test_mal.shape[0]
            rec_mal[folder] = model.predict(X_test_mal)
            mse_mal[folder] = np.mean(np.power(X_test_mal - rec_mal[folder], 2), axis=1)

        # The number of faulty samples for a 2% FPR (on the training set)
        rec_val = model.predict(X_val)
        mse_val = np.mean(np.power(X_val - rec_val, 2), axis=1)
        th = get_threshold(X_val, mse_val)

        print(f"[*] Day {day} calculated threshold: {th:.5f}")

        # Measure in the testset
        rec_ben = model.predict(X_test)
        mse_ben = np.mean(np.power(X_test - rec_ben, 2), axis=1)
        print(
            f"[*] Day {day} false positives day {day+1}: { 100*sum(mse_ben > th) / len(X_test):.2f}% ({sum(mse_ben > th)} out of {len(X_test)})"
        )
        anomalies_ben = sum(mse_ben > th)
        num_examples_test = X_test.shape[0]

        anomalies_mal = 0
        for folder in mal_folders:
            anomalies_mal += sum(mse_mal[folder] > th)
            print(
                f"[*] Day {day} {folder} detected: {100*sum(mse_mal[folder] > th) / len(mse_mal[folder]):.2f}% ({sum(mse_mal[folder] > th)} out of {len(mse_mal[folder])})"
            )

        accuracy, tpr, fpr = metrics(
            anomalies_ben, anomalies_mal, num_examples_test, num_malware
        )

        # Metrics on the test set for both malware and benign data
        print(f"[*] Day {day} centralized accuracy: {100*accuracy:.2f}%")
        print(f"[*] Day {day} centralized tpr: {100*tpr:.2f}%")
        print(f"[*] Day {day} centralized fpr: {100*fpr:.2f}%")


if __name__ == "__main__":
    main()
