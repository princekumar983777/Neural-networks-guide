from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import base64
import re
import io
import pickle
from model import DigitRecognition
 # Assuming model.py is in the same directory

app = Flask(__name__)

# Load your model
W1 = np.load('npVectors/W1.npy')
b1 = np.load('npVectors/b1.npy')
W2 = np.load('npVectors/W2.npy')
b2 = np.load('npVectors/b2.npy')
model = DigitRecognition(W1, b1, W2, b2)

def preprocess(image_data):
    # Remove header and decode
    image_data = re.sub('^data:image/.+;base64,', '', image_data)
    img_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(img_bytes)).convert('L')
    image = image.resize((28, 28))
    image = 255 - np.array(image)  # Invert: black digit on white background
    image = image / 255.0
    image = image.reshape(1, 784)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img = preprocess(data['image'])
    prediction = model.predict(img)
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
