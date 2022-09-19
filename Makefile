IP_ADDRESS?=localhost
PORT?=8080
LOAD_MODEL?=1
FIT_CLIENTS?=10
EVAL_CLIENTS?=10

build_anomaly_detection_docker:
	docker build -t stratosphere/feel-ad:latest -f docker/anomaly_detection/Dockerfile .

run_client:
	docker run --network=host --volume "$(DATA_DIR)":/data stratosphere/feel-ad client --client_id $(CLIENT_ID) --day $(DAY) --ip_address $(IP_ADDRESS)

run_server:
	docker run --network=host --volume "$(DATA_DIR)":/data stratosphere/feel-ad server --day $(DAY) --ip_address $(IP_ADDRESS) --load $(LOAD_MODEL) --num_fit_clients=$(FIT_CLIENTS) --num_evaluate_clients=$(EVAL_CLIENTS)