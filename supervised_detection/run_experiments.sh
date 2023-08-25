#!/bin/bash

experiments=("exp_sup_v1_True_1800s")

for exp in "${experiments[@]}"
do
  docker run --volume $(readlink -f experiments):/opt/feel/experiments ../feel-experiment-res SUP $exp
done
