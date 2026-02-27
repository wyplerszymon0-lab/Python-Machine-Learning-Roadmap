import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.rand(2, 1)
        self.bias = np.random.rand(1)
        self.learning_rate = 0.5

    def predict(self, inputs):
        dot_product = np.dot(inputs, self.weights) + self.bias
        return sigmoid(dot_product)

    def train(self, inputs, targets, iterations):
        for _ in range(iterations):
            prediction = self.predict(inputs)
            error = targets - prediction
            adjustment = error * sigmoid_derivative(prediction)
            self.weights += np.dot(inputs.T, adjustment) * self.learning_rate
            self.bias += np.sum(adjustment) * self.learning_rate

training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
training_targets = np.array([[0], [0], [0], [1]])

brain = NeuralNetwork()

print("Starting training process...")
brain.train(training_inputs, training_targets, 10000)
print("Training complete!\n")

print("Testing the Neural Network results:")
for test_input in training_inputs:
    prediction = brain.predict(test_input)
    print(f"Input: {test_input} | Prediction: {prediction[0]:.4f}")
