import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import argparse
import os
import datetime

from common.models import get_classification_model
from common.data_loading import load_all_data
from common.utils import MinMaxScaler, pprint_cm
from pathlib import Path
import fire

data_dir = Path('../data')

# tf.config.run_functions_eagerly(True)

def create_dataset(day, data_dir: Path, seed):
    return load_all_data(day, data_dir, seed=seed)

def metrics(fp, tp, num_ben, num_malware):
    # fp = anomalies_ben
    # tp = anomalies_mal
    tn = num_ben - fp
    # fn = num_malware - tp

    accuracy = (tp + tn) / (num_ben + num_malware)
    tpr = tp / num_malware
    fpr = fp / num_ben

    return accuracy, tpr, fpr


def main(day: int, seed: int, model_path: str, data_dir: str):
    # Parse command line arguments
    # parser = argparse.ArgumentParser(description="Centralized training scenario 1a")
    # parser.add_argument("--day", type=int, choices=(range(1,6)), required=True, help="Day to use for training.")
    # parser.add_argument("--seed", type=int, required=False, default=8181, help="Random state seed.")
    # parser.add_argument("--num_clients", "-n", type=int, choices=(range(1, 11)), required=False, default=10)
    # args = parser.parse_args()
    
    day = day

    tf.keras.utils.set_random_seed(seed)
    model = get_classification_model(Path(model_path), 2, encoder_lr=1e-4, classifier_lr=1e-4)
    X_train, X_val, X_test, y_train, y_val, y_test = create_dataset(day, Path(data_dir), seed)
    
    EPOCHS = 50
    BATCH_SIZE = 16

    scaler = MinMaxScaler().fit(X_train)
    X_train, X_val, X_test = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)


    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(
        x=X_train, y=y_train,
        shuffle=True,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard_callback]  
    )

    y_pred_raw = model.predict(X_test)
    test_loss = model.evaluate(X_test)

    print(y_pred_raw)
    y_pred = (y_pred_raw > 0.5).astype(float).T[0]

    report = classification_report(y_test, y_pred, target_names=['Benign', 'Malicious'], output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    num_malware = (y_test == 1).sum()

    anomalies_mask = y_pred == 1
    anomalies_true = y_test[anomalies_mask]

    fp = int((anomalies_true == 0).sum())
    tp = int((anomalies_true == 1).sum())
    tn = len(y_test) - fp

    tpr = tp / num_malware
    fpr = fp / num_malware


    # Metrics on the test set for both malware and benign data
    print(f"Centralized accuracy: {100*report['accuracy']:.2f}%")
    print(f"Centralized tpr: {100*tpr:.2f}%")
    print(f"Centralized fpr: {100*fpr:.2f}%")

    pprint_cm(conf_matrix, ['Benign', 'Malicious'])


if __name__ == "__main__":
    fire.Fire(main)
