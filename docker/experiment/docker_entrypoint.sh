#!/bin/bash

EXP_TYPE=$1
CONFIG_NAME=$2

PROJECT_DIR="$(readlink -f .)"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"

if [ $EXP_TYPE == "AD" ]
then
  cd anomaly_detection
  cp "experiment_configs/${CONFIG_NAME}.conf" "${CONFIG_NAME}.conf"
  python anomaly_detection_experiment.py "${CONFIG_NAME}.conf"
else
  cd supervised_detection
  cp "experiment_configs/${CONFIG_NAME}.conf" "${CONFIG_NAME}.conf"
  python supervised_experiment.py "${CONFIG_NAME}.conf"
fi
