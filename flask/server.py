from flask import Flask, jsonify, request
import numpy as np
import tensorflow as tf
from PIL import Image
import climage

MODEL_PATH = 'model_repository/mnist_model/1/model.savedmodel/'

app = Flask(__name__)
app.config['DEBUG'] = False

# Load the Model
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/mnist_infer', methods=['POST'])
def hand_classifier():
    # Receive the encoded frame and convert it back to a Numpy Array
    file = request.files['image']

    image = Image.open(file.stream)
    
    # Reshape the image to be a 28x28 array
    image = image.resize((28, 28))
    # Convert the image to a numpy array
    image = np.array(image)

    image = image.reshape(1, 28, 28, 1) # Add dimensions to create appropriate tensor shapes

    # Run inference on the frame
    hand = model.predict(image)

    print("="*50)
    output = climage.convert(file)
    print(output)
    print("Prediction: " + str(np.argmax(hand)))
    print("="*50, end='\n\n')

    return {"Prediction": str(np.argmax(hand))} # Because only string can be converted to JSON


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', threaded=True, use_reloader=True)

    # curl -X POST -F image=@images/sample_image.png http://127.0.0.1:5000/mnist_infercurl -X POST -F image=@images/sample_image.png http://127.0.0.1:5000/mnist_infer