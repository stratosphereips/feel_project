# FEEL project
Code repository for the FEEL project

# Goal
Detect malicious SSL/TLS traffic using federated learning.

# Docker setup 
You can build docker images of the client to run experiment. 
To build the image simply run 

```bash
make build_client_docker
```

and to run it you cane either set environmental variables and use also use make

```bash
CLIENT_ID=1 DAY=1 IP_ADDRESS=127.0.0.1 DATA_DIR="$(pwd)/data/" make run_client
```

or run it directly

```bash
docker run --volume "$(pwd)/data/":/data stratosphere/feel-client --client_id 1 --day 1 --ip_address 127.0.0.1
```
This way you can also specify additional arguments such as `--port` or `--seed`
