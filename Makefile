build_client_docker:
	docker build -t stratosphere/feel-client:latest -f docker/client/Dockerfile .

run_client:
	docker run --volume "$(DATA_DIR)":/data stratosphere/feel-client --client_id $(CLIENT_ID) --day $(DAY) --ip_address $(IP_ADDRESS)

fetch_dataset:
	wget -O - https://github.com/stratosphereips/feel_data/raw/main/features/data.tar.gz | tar xvz -C .