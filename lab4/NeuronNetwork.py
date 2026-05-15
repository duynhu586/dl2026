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