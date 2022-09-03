# MNIST inference on NVIDIA Triton

## Flask (Custom) Model Server
We need a model. So, let’s train one! Feel free to use any compatible model, I am using the official Keras MNIST example, and save the model in TensorFlow SavedModel format.
- Run the training script to train and save the model.
```sh
$ python train.py
```

#### 📄NOTE - The Triton server expects the models and their metadata to be arranged in a specific format. The following is an example for the TensorFlow and PyTorch model, respectively:
```
tfmobilenet
├── 1
│   └── model.savedmodel
│       └── serialized files
├── config.pbtxt
└── labels.txt
torchmobilenet
├── 1
│   └── model.pt
├── config.pbtxt
└── labels.txt
```

#### So, 
After training we need to restructure the model contents to load model into `Triton Inference Server` in the next section.<br>
Move the SavedModel contents to `<model-name>/1/model.savedmodel/<SavedModel-contents>`. The detailed instructions can be found in the official readme provided by Nvidia.
```
.
├── mnist_model
│   ├── 1
│   │   └── model.savedmodel
│   │       ├── assets
│   │       ├── keras_metadata.pb
│   │       ├── saved_model.pb
│   │       └── variables
│   │           ├── variables.data-00000-of-00001
│   │           └── variables.index
... ... ...
... ... ...
```


- Run inference script which is integrated with Flask to host it as a server which will act as the “model server”.
```sh
$ python flask/app.py
```
- Test the model server using curl.
```sh
$ curl -X POST -F image=@images/sample_image.png http://127.0.0.1:5000/mnist_infer
```

## Triton Inference Server
#### Install Triton Docker Image

Before you can use the Triton Docker image you must install
[Docker](https://docs.docker.com/engine/install). If you plan on using
a GPU for inference you must also install the [NVIDIA Container
Toolkit](https://github.com/NVIDIA/nvidia-docker).
Pull the image using the following command:

```sh
$ docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3
```

Where \<xx.yy\> is the version of Triton that you want to pull. I am using `22.08` currently.
