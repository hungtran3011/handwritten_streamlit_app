import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import cv2

class SimpleNN:
    def __init__(self, input_size, hidden_size_layer1, hidden_size_layer2=None, hidden_size_layer3=None, output_size=1, learning_rate=0.01, l2_lambda=0.001):
        self.learning_rate = learning_rate
        self.layers = []
        self.A_prev = []  # Dùng để lưu các kết quả
        self.l2_lambda = l2_lambda

        # Khởi tạo trọng số với Xavier Initialization
        self.layers.append(np.random.randn(input_size, hidden_size_layer1) * np.sqrt(2 / (input_size + hidden_size_layer1)))

        if hidden_size_layer2 is not None:
            self.layers.append(np.random.randn(hidden_size_layer1, hidden_size_layer2) * np.sqrt(2 / (hidden_size_layer1 + hidden_size_layer2)))

        if hidden_size_layer3 is not None:
            self.layers.append(np.random.randn(hidden_size_layer2 if hidden_size_layer2 is not None else hidden_size_layer1, hidden_size_layer3) * np.sqrt(2 / ((hidden_size_layer2 if hidden_size_layer2 is not None else hidden_size_layer1) + hidden_size_layer3)))

        self.W_output = np.random.randn((hidden_size_layer3 if hidden_size_layer3 is not None else (hidden_size_layer2 if hidden_size_layer2 is not None else hidden_size_layer1)), output_size) * np.sqrt(2 / ((hidden_size_layer3 if hidden_size_layer3 is not None else (hidden_size_layer2 if hidden_size_layer2 is not None else hidden_size_layer1)) + output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def soft_max(self, z):
        z_max = np.max(z, axis=1, keepdims=True)  # Tìm giá trị lớn nhất
        e_z = np.exp(z - z_max)  # Trừ giá trị lớn nhất
        return e_z / np.sum(e_z, axis=1, keepdims=True)  # Tính softmax

    def loss_function(self, output, y):
        return -np.sum(y * np.log(output)) / y.shape[0] + self.l2_lambda * np.sum([np.sum(np.square(layer)) for layer in self.layers]) + self.l2_lambda * np.sum(np.square(self.W_output))

    def forward(self, X):
        self.A = X
        self.A_prev = []

        for W in self.layers:
            self.Z = np.dot(self.A, W)
            self.A = self.sigmoid(self.Z)
            self.A_prev.append(self.A)

        self.Z_output = np.dot(self.A, self.W_output)
        self.A_output = self.soft_max(self.Z_output)
        return self.A_output

    def backward(self, X, y):
        output_delta = self.A_output - y
        self.W_output -= (self.A.T.dot(output_delta) + 2 * self.l2_lambda * self.W_output) * self.learning_rate

        hidden_delta = output_delta.dot(self.W_output.T) * self.sigmoid_derivative(self.A_prev[-1])

        for i in reversed(range(len(self.layers))):
            if i == 0:
                self.layers[i] -= (X.T.dot(hidden_delta) + 2 * self.l2_lambda * self.layers[i]) * self.learning_rate
            else:
                self.layers[i] -= (self.A_prev[i - 1].T.dot(hidden_delta) + 2 * self.l2_lambda * self.layers[i]) * self.learning_rate

            if i > 0:
                hidden_delta = hidden_delta.dot(self.layers[i].T) * self.sigmoid_derivative(self.A_prev[i - 1])

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)

    def evaluate(self, X, y):
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        loss = self.loss_function(output, y)
        return accuracy, loss
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Load your .pkl model
model = joblib.load('best_simple_nn_model.pkl')

st.title('Handwritten Digit Recognition')

# Allow user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    
    # Convert the image to grayscale and resize it to 28x28
    image = image.convert('L')
    image = np.array(image)
    image = cv2.resize(image, (28, 28))
    image = image.reshape(1, -1)

    # Normalize the image
    image = image / 255.0

    # make the image black and white
    image[image < 0.5] = 0

    # # remove image noise
    # image[image >= 0.5] = 1
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # make the image high contrast
    image = 1 - image

    #show the image to the user
    st.image(image.reshape(28, 28), caption='Resized Image.', use_container_width=True)

    # Make a prediction
    prediction = model.predict(image)
    print(prediction)

    st.write(f'Predicted Digit: {prediction[0]}')
