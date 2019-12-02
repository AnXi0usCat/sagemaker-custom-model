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

#COPY train_model.py /opt/ml/train_model.py
#COPY rest_api.py /opt/ml/rest_api.py
#COPY iris.csv /opt/ml/iris.csv

COPY train.py /opt/ml/train.py
COPY serve.py /opt/ml/serve.py

WORKDIR /opt/ml

ENTRYPOINT ["python3.6", "/opt/ml/train.py"]
