#!/bin/bash

docker rm -fv ignacio_event_datasets

docker run -it --gpus '"device=0"' --name ignacio_event_datasets -v /home/ignacio.bugueno/cachefs/datasets/:/app/ ignacio_event_datasets
