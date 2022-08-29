#!/bin/bash

# docker run --gpus all --rm -it -p 8000-8002:8000-8002 --name triton_mnist -v ${PWD}:/models nvcr.io/nvidia/tritonserver:xx.yy-py3 tritonserver --model-repository=/models --strict-model-config=false
docker run --gpus all --rm -it -p 8000-8002:8000-8002 --name triton_mnist -v ${PWD}:/models nvcr.io/nvidia/tritonserver:22.08-py3 tritonserver --model-repository=/models --strict-model-config=false

