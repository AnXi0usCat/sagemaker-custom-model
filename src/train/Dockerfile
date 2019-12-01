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

COPY lightgbm.py /opt/ml/lightgbm.py
COPY rest_api.py /opt/ml/rest_api.py

ENTRYPOINT ["/usr/bin/python3", "/opt/ml/lightgbm.py"]
