import numpy as np

np.random.seed(42)

n = 2000
d_signal = 25
d_noise = 200

X_signal = np.random.randn(n, d_signal)
X_noise = np.random.randn(n, d_noise)

w1 = np.random.randn(d_signal)
w2 = np.random.randn(d_signal)
w3 = np.random.randn(d_signal)

z = np.sin(X_signal @ w1) + 0.5 * np.sin(X_signal @ w2) + 0.3 * (X_signal @ w3)

y = (z > 0).astype(int)

flip_mask = np.random.rand(n) < 0.25
y[flip_mask] = 1 - y[flip_mask]

X = np.hstack((X_signal, X_noise))

np.savez("init_dataset_highdim.npz", X=X, y=y)