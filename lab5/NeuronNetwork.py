import math
import random
from Layer import Layer


class NeuronNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, Layer): 
        self.layers.append(Layer)

    def init_random(self, layer_sizes):
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i + 1])
            for j in range(layer_sizes[i + 1]):
                weights = [random.uniform(-1, 1) for _ in range(layer_sizes[i])]
                bias = random.uniform(-1, 1)
                layer.add_node(weights, bias)
            self.add_layer(layer)

    def init_file(self, files_path):
        lines = self.read_file(files_path)
        N, sizes = self.get_architecture(lines)
        self.build_layer(lines, N, sizes)

    def read_file(self, files_path):
        with open(files_path, 'r') as f:
            lines = []
            for line in f:
                lines.append(line.strip())
        return lines
    
    def get_architecture(self, lines):
        N = int(lines[0])
        sizes = [int(lines[i]) for i in range(1, N + 1)]
        return N, sizes
    
    def build_layer(self, lines, N, sizes):
        current_line = N + 1
        for i in range(N - 1):
            layer = Layer(sizes[i + 1])
            for j in range(sizes[i + 1]):
                parts = lines[current_line].split()
                weights = list(map(float, parts[:-1]))
                bias = float(parts[-1])
                layer.add_node(weights, bias)
                current_line += 1
            self.add_layer(layer) 

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def forward_train(self, inputs):
        for layer in self.layers:
            inputs = layer.forward_train(inputs)
        return inputs
        
    def loss(self, predict, y):
        return -y * math.log(predict) - (1 - y) * math.log(1 - predict)
    
    def loss_all(self, predicts, y):
        sum = 0
        for i in range(len(y)):
            sum += self.loss(predicts[i], y[i])
        return sum / len(y)
    
    def backprop(self, y):
        for node in self.layers[-1].nodes:
            node.delta = node.output - y
 
        for l in range(len(self.layers) - 2, -1, -1):
            curr_layer = self.layers[l]
            next_layer = self.layers[l + 1]
            for i in range(len(curr_layer.nodes)):
                node = curr_layer.nodes[i]
                grad = 0

                for next_node in next_layer.nodes:
                    grad += next_node.delta * next_node.weights[i]

                node.delta = grad * node.sigmoid_derivative()
    
    def train(self, X, y, lr=0.6, threshold=0.2, max_epochs=10000):
            predictions = []

            for sample in X:
                pred = self.forward(sample)[0]
                predictions.append(pred)

            current_loss = self.loss_all(predictions, y)
    
            for epoch in range(1, max_epochs + 1):
                grad_w = []   
                grad_b = [] 
    
                for layer in self.layers:
                    gw_layer = [[0.0] * len(node.weights) for node in layer.nodes]
                    gb_layer = [0.0] * len(layer.nodes)
                    grad_w.append(gw_layer)
                    grad_b.append(gb_layer)
    
                for i in range(len(y)):
                    self.forward_train(X[i])
                    self.backprop(y[i])
    
                    for l, layer in enumerate(self.layers):
                        for j, node in enumerate(layer.nodes):
                            for k in range(len(node.weights)):
                                grad_w[l][j][k] += node.delta * node.inputs[k]
                            grad_b[l][j] += node.delta
    
                n = len(y)
    
                for l in range(len(self.layers)):
                    layer = self.layers[l]

                    for j in range(len(layer.nodes)):
                        node = layer.nodes[j]

                        dw = []
                        for k in range(len(node.weights)):
                            dw.append(grad_w[l][j][k] / n)

                        db = grad_b[l][j] / n

                        node.weights_update(dw, db, lr)
    
                current_loss = self.loss_all(predictions, y)
                # print(f"{epoch:>6}  {current_loss:>10.6f}")
    
                if current_loss < threshold:
                    # print(f"\n epoch = {epoch}, loss = {current_loss:.6f}")
                    return epoch
    
            print(f"\nmax epochs = {max_epochs}, loss_final = {current_loss:.6f}")
            return max_epochs
        