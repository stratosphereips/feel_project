#!/bin/bash

normal_dir=$1
out_dir=$2
extractor_path=$3
tw=$4

normal_folders=( `ls $normal_dir` )
for normal_folder in "${normal_folders[@]}"
do
  normal_day_folders=( `ls -d ${normal_dir}/${normal_folder}/*/` )
  for day_folder in "${normal_day_folders[@]}"
  do
    ./generate_features.sh ${day_folder} $out_dir $extractor_path $tw
  done
done
