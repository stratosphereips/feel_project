FROM tensorflow/tensorflow:2.9.1

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PYTHONHASHSEED=random PYTHONDONTWRITEBYTECODE=1

#EXPOSE 8080

ENV MOUNT_DATA_DIR=/data
ENV WORK_DIR=/opt/feel

RUN mkdir -p "$MOUNT_DATA_DIR" &&\
    mkdir $WORK_DIR

RUN wget -c\
    https://github.com/stratosphereips/feel_data/blob/main/features/data.tar.gz?raw=true\
    -nv -O - | tar -xz -C ${MOUNT_DATA_DIR}/..

VOLUME $MOUNT_DATA_DIR

WORKDIR /opt/feel

COPY anomaly_detection $WORK_DIR
COPY requirements.txt $WORK_DIR

# Do not install tensorflow, it's already in the image
RUN sed -i 's/tensorflow/#tensorflow/g' $WORK_DIR/requirements.txt

RUN python -m pip install --upgrade pip
RUN python -m pip install -r $WORK_DIR/requirements.txt

ENTRYPOINT ["./docker_entrypoint.sh"]




