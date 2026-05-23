from Node import Node

class Layer:
    def __init__(self, num_nodes):
        self.nodes = []
        self.num_nodes = num_nodes

    def add_node(self, weights, bias):
        node = Node(weights, bias)
        self.nodes.append(node)

    def forward(self, inputs):
        outputs = []
        for node in self.nodes:
            outputs.append(node.activation(inputs))
        return outputs

    def forward_train(self, inputs):
        outputs = []
        for node in self.nodes:
            outputs.append(node.forward_train(inputs))
        return outputs