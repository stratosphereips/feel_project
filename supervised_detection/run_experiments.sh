#!/bin/bash

experiments=("exp_sup_v1_False" "exp_sup_v1_False_v2_True" "exp_sup_v1_False_v3_False" "exp_sup_v1_False_v4_scenario" "exp_sup_v1_True" "exp_sup_v1_True_v2_True" "exp_sup_v1_True_v3_False" "exp_sup_v1_True_v4_scenario" )

for epx in "${experiments[@]}"
do
  docker run -f --volume $(readlink -f experiments):/opt/feel/experiments janatpa/feel-experiment SUP exp
done

( "exp_sup_v1_False_v2_False_v4_scenario" "exp_sup_v1_True_v2_False_v4_scenario" "exp_sup_v1_True_v2_True_v4_scenario"