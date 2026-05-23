from NeuronNetwork import NeuronNetwork
from Layer import Layer

xor = NeuronNetwork()
xor.init_random([2, 3, 1])

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

print("xor")
xor.train(x, y, max_epochs=1000, lr=0.5, threshold=0.01)
print("xor.forward(x[0]):", xor.forward(x[0]))
print("xor.forward(x[1]):", xor.forward(x[1]))
print("xor.forward(x[2]):", xor.forward(x[2]))
print("xor.forward(x[3]):", xor.forward(x[3]))

def loadfunc(filestr):
    with open(filestr, 'r') as f:
        results = []
        for line in f:
                words = line.split(',')
                results.append((words[0], words[1].rstrip("\n\r"), words[2].rstrip("\n\r")))
    return results

data = loadfunc('loan2.csv')

x = []
y = []

for i in range(len(data) - 1):
    features = [
        float(data[i + 1][0]),
        float(data[i + 1][1])
    ]

    target = float(data[i + 1][2])

    x.append(features)
    y.append(target)

nn = NeuronNetwork()

nn.init_random([2,1])

nn.train(x, y, max_epochs=1000, lr=0.5, threshold=0.01)

print("nn.forward(x[2]):", nn.forward(x[2]))