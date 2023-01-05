#!/bin/bash

day=$1
num_clients=$2

PROJECT_DIR="$(readlink -f ..)"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"

./certificates/generate.sh

echo "Starting server for day ${day}, seed ${seed}, and ${num_clients} clients."
python server.py --day ${day} --config_path exp_sup_v1_False_v4_scenario.conf &

sleep 5  # Sleep for 3s to give the server enough time to start
for i in `seq 1 ${num_clients}`; do
    echo "Starting client $i"
    python client.py --day ${day} --client_id ${i} --config_path exp_sup_v1_False_v4_scenario.conf &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
