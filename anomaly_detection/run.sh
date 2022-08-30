#!/bin/bash

day=$1
seed=$2

./certificates/generate.sh

# Clean the previous saved data if they exist
rm *.npz

echo "Starting server"
python server.py --day=${day} --seed=${seed} --load=0 --data_dir="/opt/Malware-Project/BigDataset/FEELScenarios/"&
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 1 10`; do
    echo "Starting client $i"
    python client.py --day=${day} --client_id=${i} --seed=${seed} --data_dir="/opt/Malware-Project/BigDataset/FEELScenarios/"&
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
