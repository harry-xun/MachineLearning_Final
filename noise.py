import numpy as np

data = np.load("init_dataset.npz")
X = data["X"]
y = data["y"]

w = np.array([1.0, -0.5])
b = -2.5
dist = np.abs(X @ w + b) / np.linalg.norm(w)

flipped = 0
constant = 2.2
y_noisy = y.copy()
for i in range(len(y)):
    flip_prob = np.exp(-constant * dist[i])
    r = np.random.rand()
    if r < flip_prob:
        y_noisy[i] = 1 - y_noisy[i]
        flipped += 1
print("flipped:", flipped)
np.savez("noise_dataset.npz", X=X, y=y_noisy)