IP_ADDRESS?=localhost
PORT?=8080
LOAD_MODEL?=1
FIT_CLIENTS?=10
EVAL_CLIENTS?=10
OUT_DIR?=experiments
EXPERIMENT_TYPE?=sup

build_anomaly_detection_docker:
	docker build -t stratosphere/feel-ad:latest -f docker/anomaly_detection/Dockerfile .

build_experiment_docker:
	docker build -t janatpa/feel-experiment:latest -f docker/experiment/Dockerfile .

run_client:
	docker run --network=host --volume "$(DATA_DIR)":/data stratosphere/feel-ad client --client_id $(CLIENT_ID) --day $(DAY) --ip_address $(IP_ADDRESS)

run_experiment:
	docker run --volume $$(readlink -f $(OUT_DIR)):/opt/feel/experiments stratosphere/feel-experiment $(EXPERIMENT_TYPE) $(EXPERIMENT_NAME)

fetch_dataset:
	wget -O - https://github.com/stratosphereips/feel_data/raw/main/features/data.tar.gz | tar xvz -C .

run_server:
	docker run --network=host --volume "$(DATA_DIR)":/data stratosphere/feel-ad server --day $(DAY) --ip_address $(IP_ADDRESS) --load $(LOAD_MODEL) --num_fit_clients=$(FIT_CLIENTS) --num_evaluate_clients=$(EVAL_CLIENTS)
