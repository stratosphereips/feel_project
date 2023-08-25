#!/bin/bash

experiments=("exp_sup_v1_True_v2_True_1800s_2d_scenario" "exp_sup_v1_True_v2_True_1800s_3d_scenario" "exp_sup_v1_True_v2_True_1800s_scenario")

PROJECT_DIR="$(readlink -f ..)"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"

for exp in "${experiments[@]}"
do
  cp "experiment_configs/${exp}.conf" "${exp}.conf"
  python supervised_experiment.py "${exp}.conf"
done
 $HOME/.profile