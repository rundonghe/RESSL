import numpy as np

def analyze_data(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x * y)
    sum_x_squared = sum(x * x)
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)

    avg = np.mean(y)
    GM = np.sum(np.abs(y - avg))

    s = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    ma = np.max(s)
    mi = np.min(s)

    P = np.sum(s >= 0) / len(s)

    return slope, GM, ma, mi, P


x = np.array([0,0.5,1])
y = np.array([0.394, 0.391, 0.393])

R, GM, BAD, WAD, P = analyze_data(x, y)
print("R:", R)
print("GM:", GM)
print("BAD:", BAD)
print("WAD:", WAD)
print("P:", P)
