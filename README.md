# FEEL project

The feel project aims at researching, developing and implementing a federated learning ML model for intrusion detection systems.
It was implemented by Pavel Janata's and published for his thesis "Decentralized Federated Learning for Network Security".

This project is funded by [NlNet NGI Zero Entrust](https://nlnet.nl/project/Iris-P2P/)

# Goal
Detect malicious SSL/TLS traffic using federated learning.

# How it works
We use horizontal cross-device federated learning for detecting malicious activity in encrypted TLS network traffic. Cross-device in this context means, that the clients represent edge computers, monitoring and capturing their traffic. It is horizontal because the clients observe the same set of features, produced by different entities. 

The federated approach allows to distributively train a model using the client’s observations, without having direct access to the data. This enables us to protect the privacy of the data, while still being able to learn from it. In addition, each client also benefits from cooperative training, as they use a global detection model that is averaged from all model updates sent by all the clients. The global model, therefore, had access to a larger and more diverse set of data coming from all clients, possibly leading to better performance and generalization, compared to a model trained only with each client's local data.


# Docs

For more details about how it works you can check the docs at 
https://github.com/stratosphereips/StratosphereLinuxIPS/blob/develop/docs/feel_project.md

And the thesis of Pavel Janata: [Decentralized Federated Learning for Network Security](https://dspace.cvut.cz/bitstream/handle/10467/107647/F3-DP-2023-Janata-Pavel-Master_Thesis_Pavel_Janata.pdf)

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

# Related projects

This project is now a submodule of [StratosphereLinuxIPS](https://github.com/stratosphereips/StratosphereLinuxIPS)


