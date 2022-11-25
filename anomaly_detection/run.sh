#!/bin/bash

day=$1
num_clients=$2

PROJECT_DIR="$(readlink -f ..)"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"

#./certificates/generate.sh

# Clean the previous saved data if they exist
rm *.npz

echo "Starting server for day ${day}, seed ${seed}, and ${num_clients} clients."
python server.py --day ${day} --num_fit_clients ${num_clients} --num_evaluate_clients ${num_clients} &

sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 1 ${num_clients}`; do
    echo "Starting client $i"
    python client.py --day ${day} --client_id ${i} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
