#!/bin/bash

exp_names=( exp_sup_v1_False exp_sup_v1_True exp_sup_v1_True_v2_True_v4_scenario exp_sup_v1_False_v2_True_v4_scenario )
exp_types=( SUP SUP SUP SUP )

idx=0
length=${#exp_names[@]}

while [[ $idx < $length ]]
do
  if [[ $(docker ps | grep feel | wc -l) != 3 ]]
  then
    exp_name=${exp_names[$idx]}
    exp_type=${exp_types[$idx]}
    echo "Running ${exp_name}"
    docker run -d --rm --user pavel --name "feel_${exp_name}" --volume $(readlink -f experiments):/opt/feel/experiments janatpa/feel-experiment $exp_type $exp_name &
    idx=$(($idx+1))
    sleep 60
  else
    echo "Sleeping for 30 seconds"
    sleep 30
  fi
done
