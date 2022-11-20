#!/bin/bash

day=$1
seed=$2
num_clients=$3
model_path=$4

PROJECT_DIR="$(readlink -f ..)"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"

./certificates/generate.sh

echo "Starting server for day ${day}, seed ${seed}, and ${num_clients} clients."
python server.py --day ${day} &

sleep 3  # Sleep for 3s to give the server enough time to start
for i in 1 2 3 4 5 6 7 8 9 10; do
    echo "Starting client $i"
    python client.py --day ${day} --client_id ${i} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
