#!/bin/bash

in_dir=$1
out_dir=$2
extractor_path=$3
tw=$4

if [[ $# -lt 4 ]] ; then
    echo 'Missing input arguments. Usage:'
    echo './script.sh <input_dir> <output_dir> <extractor_dir> <time_window>'
    exit 1
fi


# Split the conn.log.labeled into time windows 
awk -v inc=$tw -v out_dir=$out_dir -f split_conn.awk $in_dir/conn.log.labeled

# Get all the genrated folders and copy the ssl and x509 logs
folders=( `ls $2` )
for folder in "${folders[@]}"
do
   cp $in_dir/x509.log $out_dir/$folder
   cp $in_dir/ssl.log.labeled $out_dir/$folder
   python $extractor_path/feature_extractor.py -z $out_dir/$folder
done

# Combine all the CSVs that are in the extracted directory
# and store them in the original input directory along with the los
python combine_features.py -i $out_dir -o $in_dir 

# Optional better to do it manually in case something bad happens :)
# rm -rf $2/* 