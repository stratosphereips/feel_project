import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import argparse
import os

from utils import get_model, get_threshold

data_dir = '/opt/Malware-Project/BigDataset/FEELScenarios/'


def create_dataset():
    data = dict()
    for j in range(1, 6):
        data["Day"+str(j)] = pd.DataFrame()
        for i in range(1, 11):
            df_temp = pd.read_csv(os.path.join(data_dir, 'Processed', 'Client'+str(i), 'Day'+str(j), "comb_features_ben.csv"))
            data["Day"+str(j)] = pd.concat([data["Day"+str(j)], df_temp], ignore_index=True)
    

    for day in range(1, 6):
        data["Day"+str(day)] = data["Day"+str(day)].drop(["ssl_ratio", "self_signed_ratio", "SNI_equal_DstIP", "ratio_certificate_path_error", "ratio_missing_cert_in_cert_path"], axis=1)
        data["Day"+str(day)] = data["Day"+str(day)].drop_duplicates()


    mal_data = dict()
    mal_folders = ['CTU-Malware-Capture-Botnet-346-1', 'CTU-Malware-Capture-Botnet-327-2', 'CTU-Malware-Capture-Botnet-230-1', 'CTU-Malware-Capture-Botnet-219-2']

    for folder in mal_folders:
        mal_data[folder] = pd.DataFrame()
        df_temp = pd.read_csv(os.path.join(data_dir, 'Raw', 'Malware', folder, 'Day1', "comb_features.csv"))
        mal_data[folder] = pd.concat([mal_data[folder], df_temp], ignore_index=True)
    
    # Drop columns and possible duplicates
    for folder in mal_folders:
        mal_data[folder] = mal_data[folder].drop(["ssl_ratio", "self_signed_ratio", "SNI_equal_DstIP", "ratio_certificate_path_error", "ratio_missing_cert_in_cert_path"], axis=1)
        mal_data[folder] = mal_data[folder].drop_duplicates()

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
    parser.add_argument("--day", type=int, choices=(range(1,6)), required=True, help="Day to use for training.")
    parser.add_argument("--seed", type=int, required=False, default=8181, help="Random state seed.")
    args = parser.parse_args()
    
    day = args.day

    model = get_model() 
    data, mal_data = create_dataset()
    mal_folders = list(mal_data.keys())
    
    EPOCHS = 8
    BATCH_SIZE = 64

    scaler = MinMaxScaler()
    X = scaler.fit_transform(data["Day"+str(day)])
    X_test = scaler.transform(data["Day"+str(day+1)])

    X_train , X_val = train_test_split(X, test_size=0.2, random_state=args.seed)

    model.fit(
        X_train, X_train,
        shuffle=True,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, X_val)  
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

    print(f"Calculated threshold: {th:.5f}")

    # Measure in the testset
    rec_ben = model.predict(X_test)
    mse_ben = np.mean(np.power(X_test - rec_ben, 2), axis=1)
    print(f'False positives on next day: { 100*sum(mse_ben > th) / len(X_test):.2f}% ({sum(mse_ben > th)} out of {len(X_test)})')
    anomalies_ben = sum(mse_ben > th)
    num_examples_test = X_test.shape[0]

    anomalies_mal = 0
    for folder in mal_folders:
        anomalies_mal += sum(mse_mal[folder] > th)
        print(f'{folder} detected: {100*sum(mse_mal[folder] > th) / len(mse_mal[folder]):.2f}% ({sum(mse_mal[folder] > th)} out of {len(mse_mal[folder])})')

    accuracy, tpr, fpr = metrics(anomalies_ben, anomalies_mal, num_examples_test, num_malware)

    # Metrics on the test set for both malware and benign data
    print(f"Centralized accuracy: {100*accuracy:.2f}%")
    print(f"Centralized tpr: {100*tpr:.2f}%")
    print(f"Centralized fpr: {100*fpr:.2f}%")

if __name__ == "__main__":
    main()
