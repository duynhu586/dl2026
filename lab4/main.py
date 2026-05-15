from NeuronNetwork import NeuronNetwork
from Layer import Layer

xor_nn = NeuronNetwork()

hidden = Layer(2)
hidden.add_node([-1, -1], 1.5) 
hidden.add_node([1, 1], -0.5)   
xor_nn.add_layer(hidden)

output = Layer(1)
output.add_node([1, 1], -1.5)  
xor_nn.add_layer(output)

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
print("Manual initialization:")
for i in inputs:
    res = xor_nn.forward(i)
    print(f"Input {i} -> Output {res}")

xor_file = NeuronNetwork()
xor_file.init_file('text.txt')
print("Initialization from file:")
for i in inputs:
    res = xor_file.forward(i)
    print(f"Input {i} -> Output {res}")    

xor_random = NeuronNetwork()
xor_random.init_random([2, 3, 1])
print("Random initialization:")
for i in inputs:
    res = xor_random.forward(i)
    print(f"Input {i} -> Output {res}")

