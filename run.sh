#!/bin/bash

# GPU Command
docker run --gpus all --rm -it -p 8000-8002:8000-8002 --name triton_mnist -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:22.08-py3 tritonserver --model-repository=/models --strict-model-config=false


# CPU Command 
docker run --rm -it -p 8000-8002:8000-8002 --name triton_mnist -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:22.08-py3 tritonserver --model-repository=/models --strict-model-config=false

