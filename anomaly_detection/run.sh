#!/bin/bash

day=$1
seed=$2

./certificates/generate.sh

rm *.npz

echo "Starting server"
python server.py --day=${day} --seed=${seed} --load=1&
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 1 10`; do
    echo "Starting client $i"
    python client.py --day=${day} --client_id=${i} --seed=${seed}&
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
