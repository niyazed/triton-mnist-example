import requests
import numpy as np
import tensorflow as tf
from PIL import Image
import climage
import tritonclient.http as httpclient




def preprocess(image):
    image = image.resize((28, 28))
    image = np.array(image).astype(np.float32)
    image = image.reshape(1, 28, 28, 1)

    return image

def postprocess_output(preds):
    return np.argmax(np.squeeze(preds))


if __name__ == '__main__':

    image = Image.open('images/sample_image.png')
    image = preprocess(image)


    # Create the inference context for the model.
    model_name = "mnist_model"
    model_version = 1
    triton_client = httpclient.InferenceServerClient(url="localhost:8000")
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput('input_1', [1, 28, 28, 1], "FP32"))
    inputs[0].set_data_from_numpy(image)
    outputs.append(httpclient.InferRequestedOutput('output_1'))
    results = triton_client.infer(model_name, inputs, outputs=outputs)
    output = np.squeeze(results.as_numpy('output_1'))
    pred = postprocess_output(output)

    print(pred)











    # inference tritonclient grpc request

    import tritonclient.grpc as grpcclient

    # Create the inference context for the model.
    # model_name = "mnist_model"
    # model_version = 1
    # triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
    # inputs = []
    # outputs = []
    # inputs.append(grpcclient.InferInput('input_1', [1, 28, 28, 1], "FP32"))
    # inputs[0].set_data_from_numpy(image)
    # outputs.append(grpcclient.InferRequestedOutput('output_1'))
    # results = triton_client.infer(model_name, inputs, outputs=outputs)
    # print(results)

