#!/bin/sh

docker run -v $(pwd)/testing:/opt/ml -p 8080:8080  light-gbm-container python3.6 /opt/program/serve.py 
