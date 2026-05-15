import math

class Node:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def linear_sum(self, inputs):
        self.inputs = inputs
        sum = self.bias
        for i in range(len(self.weights)):
            sum += self.weights[i] * inputs[i]
        return sum
    
    def step(self, x):
        return 1 if x >= 0 else 0
    
    def activation(self, inputs):
        sum = self.linear_sum(inputs)
        return self.step(sum)