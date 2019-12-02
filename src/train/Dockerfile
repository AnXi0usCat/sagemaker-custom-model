FROM ubuntu:16.04

RUN apt-get update && \
  apt-get install -y --no-install-recommends  software-properties-common \
  wget && \
  add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update

RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip

# update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel

# install the requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# this for testing locally
RUN bash -c 'mkdir -p /opt/ml/{input, output, model}'

COPY train.py /opt/ml/train.py
COPY serve.py /opt/ml/serve.py
COPY hyperparameters.json opt/ml/input/config/hyperparameters.json
COPY training /opt/ml/input/data/training/training.csv
COPY validation /opt/ml/input/data/validation/validation.csv

RUN mkdir /opt/ml/output && \
    mkdir /opt/ml/model  && \
    mkdir /opt/ml/output/failure

WORKDIR /opt/ml

ENTRYPOINT ["python3.6", "/opt/ml/train.py"]
