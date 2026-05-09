import matplotlib.pyplot as plt

def loadfunc(filestr):
    with open(filestr, 'r') as f:
        results = []
        for line in f:
                words = line.split(',')
                results.append((words[0], words[1].rstrip("\n\r"), words[2].rstrip("\n\r")))
    return results



data = loadfunc('loan2.csv')
x1 =[]
x2 =[]
y=[]

print(data)

col_names = data[0]
print(col_names)

for i in range(len(data) - 1):
        x1.append(float(data[i + 1][0]))
        x2.append(float(data[i + 1][1]))
        y.append(float(data[i + 1][2]))

w0 = -0.324
w1 = 0.846
w2 = -0.090

def draw_line(x):
    return -(w1 * x + w0) / w2

xpoints1 = [0.5,2]

ypoints = [draw_line(xpoints1[0]), draw_line(xpoints1[1])]

print(xpoints1)
print(ypoints)

plt.plot(xpoints1, ypoints)
plt.scatter(x1, x2, c=y, marker='o', linewidth=2)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Sample Line Plot")
plt.grid(True)
plt.show()
plt.imsave('output_image.png', data, cmap='viridis')