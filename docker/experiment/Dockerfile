FROM tensorflow/tensorflow:2.9.1

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PYTHONHASHSEED=random PYTHONDONTWRITEBYTECODE=1

ENV MOUNT_DATA_DIR=/data
ENV WORK_DIR=/opt/feel
ENV EXPERIMENT_DIR=$WORK_DIR/experiments

RUN mkdir -p "$MOUNT_DATA_DIR" &&\
    mkdir $WORK_DIR

RUN wget -c\
    https://github.com/stratosphereips/feel_data/blob/main/features/data.tar.gz?raw=true\
    -nv -O - | tar -xz -C ${MOUNT_DATA_DIR}/..

#VOLUME $MOUNT_DATA_DIR
#VOLUME $EXPERIMENT_DIR

WORKDIR /opt/feel

COPY anomaly_detection $WORK_DIR/anomaly_detection
COPY common $WORK_DIR/common
COPY experiment $WORK_DIR/experiment
COPY supervised_detection $WORK_DIR/supervised_detection
COPY requirements.txt $WORK_DIR

COPY Makefile $WORK_DIR
COPY docker/experiment/docker_entrypoint.sh .

RUN make fetch_dataset

# Do not install tensorflow, it's already in the image
RUN sed -i 's/tensorflow/#tensorflow/g' $WORK_DIR/requirements.txt

RUN python -m pip install --upgrade pip
RUN python -m pip install -r $WORK_DIR/requirements.txt

ENTRYPOINT ["./docker_entrypoint.sh"]




