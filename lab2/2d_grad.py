import matplotlib.pyplot as plt

def L(w0,w1,x,y):
    return 1/2 * ( w1*x + w0 - y)**2

def dLw0(w0, w1, x, y):
    return w1*x + w0 - y

def dLw1(w0, w1, x, y):
    return x * (w1*x + w0 - y)

def gradient_descent(x, y, w0, w1, lr, threshold):
    cnt = 0
    print(f"Time  x  f(x)")
    while abs(L(w0, w1, x, y)) > threshold:
        w0 = w0 - lr * dLw0(w0, w1, x, y)
        w1 = w1 - lr * dLw1(w0, w1, x, y)
        cnt += 1
        print(f"{cnt}  {w0:.3f} {w1:.3f} {L(w0, w1, x, y):.3f}")
    return w0, w1


def L_all(w0,w1,x,y):
    sum = 0
    for i in range(len(x)):
        sum += L(w0,w1,x[i],y[i])
    return sum/len(x)

def dLw0_all(w0, w1, x, y):
    sum = 0
    for i in range(len(x)):
        sum += dLw0(w0, w1, x[i], y[i])
    return sum/len(x)

def dLw1_all(w0, w1, x, y):
    sum = 0
    for i in range(len(x)):
        sum += dLw1(w0, w1, x[i], y[i])
    return sum/len(x)

def gradient_descent_all(x, y, w0, w1, lr, threshold):
    cnt = 0
    while abs(L_all(w0, w1, x, y)) > threshold:
        w0 = w0 - lr * dLw0_all(w0, w1, x, y)
        w1 = w1 - lr * dLw1_all(w0, w1, x, y)
        cnt += 1
        print(f"{cnt}  {w0:.3f} {w1:.3f} {L_all(w0, w1, x, y):.3f}")
    return w0, w1


def loadfunc(filestr):
    with open(filestr, 'r') as f:
        results = []
        for line in f:
                words = line.split(',')
                results.append((words[0], words[1].rstrip("\n\r")))
    return results


if __name__ == "__main__":
    data = loadfunc('lr.csv')
    x =[]
    y=[]

    for i in range(len(data)):
        x.append(float(data[i][0]))
        y.append(float(data[i][1]))
    
    learning_rate = 0.00001*3
    threshold = 10
    w0 = 0
    w1 = 1
    # x = 3
    # y = 6
    # optimal_w0, optimal_w1 = gradient_descent(x, y, w0, w1, learning_rate, threshold)
    optimal_w0, optimal_w1 = gradient_descent_all(x, y, w0, w1, learning_rate, threshold)


