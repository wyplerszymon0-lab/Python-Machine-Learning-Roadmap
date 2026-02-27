import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class MultilayerNetwork:
    def __init__(self):
        self.weights1 = np.random.rand(2, 3)
        self.weights2 = np.random.rand(3, 1)
        self.bias1 = np.random.rand(1, 3)
        self.bias2 = np.random.rand(1, 1)
        self.learning_rate = 0.5

    def predict(self, inputs):
        self.layer1 = sigmoid(np.dot(inputs, self.weights1) + self.bias1)
        self.output = sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)
        return self.output

    def train(self, inputs, targets, iterations):
        for _ in range(iterations):
            prediction = self.predict(inputs)

            error_out = targets - prediction
            delta_out = error_out * sigmoid_derivative(prediction)

            error_h = delta_out.dot(self.weights2.T)
            delta_h = error_h * sigmoid_derivative(self.layer1)

            self.weights2 += self.layer1.T.dot(delta_out) * self.learning_rate
            self.weights1 += inputs.T.dot(delta_h) * self.learning_rate
            self.bias2 += np.sum(delta_out) * self.learning_rate
            self.bias1 += np.sum(delta_h) * self.learning_rate


training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
training_targets = np.array([[0], [1], [1], [0]])

brain = MultilayerNetwork()

print("Training for XOR problem...")
brain.train(training_inputs, training_targets, 20000)
print("Training complete!\n")

print("XOR Results:")
for test_input in training_inputs:
    prediction = brain.predict(test_input)
    print(f"Input: {test_input} | Prediction: {prediction[0][0]:.4f}")
