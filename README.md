# FEEL project

The feel project aims at researching, developing and implementing a federated learning ML model for intrusion detection systems.
It was implemented by Pavel Janata's and published for his thesis "Decentralized Federated Learning for Network Security".

This project is funded by [NlNet NGI Zero Entrust](https://nlnet.nl/project/Iris-P2P/)

# Goal
Detect malicious SSL/TLS traffic using federated learning.

# Docker setup 
You can build a docker image for the anomaly detection experiment. 
To build the image, simply run 

```bash
make build_anomaly_detection_docker
```

and to run it you can either set environmental variables:

```bash
make run_client CLIENT_ID=1 DAY=1
```

or run it directly:

```bash
docker run --network-host --volume "$(pwd)/data/":/data stratosphere/feel-ad client --client_id 1 --day 1 --ip_address 127.0.0.1
```

This way you can also specify additional arguments such as `--port` or `--seed`

To run the server use: 

```bash
make run_server DAY=1
```

or directly 
```bash
docker run --network=host --volume "$(pwd)/data/":/data stratosphere/feel-ad server --day 1 --ip_address localhost --load 1 --num_fit_clients=10 --num_evaluate_clients=10
```
