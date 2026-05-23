import math

class Node:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.inputs = None
        self.output = None
        self.delta = None

    def linear_sum(self, inputs):
        self.inputs = inputs
        total = self.bias
        for i in range(len(self.weights)):
            total += self.weights[i] * inputs[i]
        return total

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self):
        return self.output * (1 - self.output)

    def activation(self, inputs):
        total = self.linear_sum(inputs)
        return self.sigmoid(total) 

    def forward_train(self, inputs):
        self.inputs = inputs
        total = self.linear_sum(inputs)
        self.output = self.sigmoid(total)
        return self.output

    def weights_update(self, dw, db, lr):
        for j in range(len(self.weights)):
            self.weights[j] -= lr * dw[j]
        self.bias -= lr * db