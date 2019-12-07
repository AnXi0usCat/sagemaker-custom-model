#!/bin/sh

rm testing/model/*
rm testing/output/*

docker run -v $(pwd)/testing:/opt/ml light-gbm-container python3.6 /opt/program/train.py
