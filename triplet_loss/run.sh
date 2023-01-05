#!/bin/bash

day=$1
seed=$2
num_clients=$3

PROJECT_DIR="$(readlink -f ..)"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"


./certificates/generate.sh

# Clean the previous saved data if they exist
rm *.npz

echo "Starting server for day ${day}, seed ${seed}, and ${num_clients} clients."
python server.py --day ${day} \
                --seed ${seed} \
                --load 1 \
                --data_dir "../data" \
                --num_clients ${num_clients} &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 1 10`; do
    echo "Starting client $i"
    python client.py --day ${day} \
                    --client_id ${i} \
                    --seed ${seed} \
		    --port 8000 \
                    --data_dir="../data"&
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
