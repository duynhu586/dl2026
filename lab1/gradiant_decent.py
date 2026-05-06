def f(x):
    return x**2

def f_(x):
    return 2*x

def gradient_descent(x, lr, threshold):
    cnt = 0
    print(f"Time  x  f(x)")
    while abs(f(x)) > threshold:
        x = x - lr * f_(x)
        cnt += 1
        print(f"{cnt}  {x:.3f} {f(x):.3f}")
    return x

if __name__ == "__main__":
    x = 10
    learning_rate = 0.1
    threshold = 0.01
    optimal_x = gradient_descent(x, learning_rate, threshold)
