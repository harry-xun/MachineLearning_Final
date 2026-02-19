import numpy as np
import matplotlib.pyplot as plt

data = np.load("init_dataset.npz")
X = data["X"]
y = data["y"]
X0 = X[y==0]
X1 = X[y==1]
plt.scatter(X0[:,0], X0[:,1])
plt.scatter(X1[:,0], X1[:,1])
plt.show()