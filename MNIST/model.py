
import numpy as np

class DigitRecognition:
    def __init__(self, W1, b1, W2, b2):
        self.input_size = 784
        self.hidden_size = 128
        self.output_size = 10
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def predict(self, X):
        z1 = X.dot(self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = a1.dot(self.W2) + self.b2
        output = self.softmax(z2)
        return np.argmax(output, axis=1)
