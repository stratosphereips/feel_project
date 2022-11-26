#!/bin/bash

EXP_TYPE=$1
CONFIG_NAME=$2

PROJECT_DIR="$(readlink -f .)"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"

if [ $EXP_TYPE == "AD" ]
then
  cd anomaly_detection
  python anomaly_detection_experiment.py "${CONFIG_NAME}.conf"
else
  cd supervised_detection
  python supervised_experiment.py "${CONFIG_NAME}.conf"
fi
