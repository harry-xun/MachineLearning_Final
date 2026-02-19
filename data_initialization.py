import numpy as np

np.random.seed(42)
n = 2000 

w = np.array([1.0, -0.5])
b = -2.5

X=[]
y=[]

for i in range(n):
    x = np.random.uniform(0, 10, 2)
    val = np.dot(w, x) + b
    if val > 0:
        label = 1
    else:
        label = 0
    X.append(x)
    y.append(label)
X = np.array(X)
y = np.array(y)
np.savez("init_dataset.npz", X=X, y=y)