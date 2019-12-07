FROM ubuntu:16.04

RUN apt-get update && \
  apt-get install -y --no-install-recommends  software-properties-common \
  nginx \ 
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


RUN mkdir /opt/ml && \
    mkdir /opt/program

COPY lightgbm /opt/program

WORKDIR /opt/program
