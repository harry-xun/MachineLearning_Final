import numpy as np
import matplotlib.pyplot as plt

data = np.load("init_dataset.npz")
X = data["X"]
y = data["y"]

X0 = X[y == 0]
X1 = X[y == 1]

plt.figure(figsize=(6,5))

plt.scatter(X0[:,0], X0[:,1], color="#6FAF8F", label="Class 0")
plt.scatter(X1[:,0], X1[:,1], color="#E06C86", label="Class 1")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Dataset distribution")

plt.legend()

plt.show()