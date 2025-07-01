from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import base64
import re
import io
from PIL import Image
from model import DigitRecognition
from tensorflow.keras.models import load_model

 # Assuming model.py is in the same directory

app = Flask(__name__)

# ANN model
W1 = np.load('npVectors/W1.npy')
b1 = np.load('npVectors/b1.npy')
W2 = np.load('npVectors/W2.npy')
b2 = np.load('npVectors/b2.npy')
ann_model = DigitRecognition(W1, b1, W2, b2)

# CNN model
cnn_model = load_model('mnsit_cnn.keras')  # or .h5

def preprocess(image_data, for_cnn=False):
    image_data = re.sub('^data:image/.+;base64,', '', image_data)
    img_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(img_bytes)).convert('L').resize((28, 28))
    image = np.array(image)  # Invert

    image = image / 255.0

    image_to_save = image.reshape(28, 28) * 255
    image_to_save = Image.fromarray(image_to_save.astype('uint8'))
    image_to_save.save('debug_ann_input.png')

    if for_cnn:
        print('called for cnn')
        return image.reshape(1, 28, 28, 1)
    else:
        print('call for ann')
        return image.reshape(1, 784)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']
    model_type = data.get('model', 'ann')  # default to 'ann'

    if model_type == 'cnn':
        img = preprocess(image_data, for_cnn=True)
        prediction = int(np.argmax(cnn_model.predict(img)))
    else:
        img = preprocess(image_data, for_cnn=False)
        prediction = int(ann_model.predict(img))

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
