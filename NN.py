import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = np.load("noise_dataset.npz")
X = data["X"]
y = data["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
T = 500
m = 200
preds = np.zeros((T, len(X_test)))
for t in range(T):
    idx = np.random.choice(len(X_train), size=m, replace=False)
    X_t = X_train[idx]
    y_t = y_train[idx]
    model = MLPClassifier(hidden_layer_sizes=(100,), activation="relu",max_iter=500, random_state=42)
    model.fit(X_t, y_t)
    preds[t] = model.predict_proba(X_test)[:, 1]
mean_pred = preds.mean(axis=0)
bias_list = []
for i in range(len(y_test)):
    diff = mean_pred[i] - y_test[i]
    bias_list.append(diff * diff)
bias2 = sum(bias_list) / len(bias_list)
var_list = []
for i in range(len(y_test)):
    s = 0
    for t in range(T):
        diff = preds[t][i] - mean_pred[i]
        s += diff ** 2
    var_i = s/T
    var_list.append(var_i)
variance = sum(var_list) / len(var_list)
mse = np.mean((preds-y_test)**2)
print("Bias^2:", bias2)
print("Variance:", variance)
print("MSE:", mse)
print("Error", bias2 + variance)





bias_arr = np.array(bias_list)
var_arr = np.array(var_list)
bias_cutoff = np.percentile(bias_arr, 90)
var_cutoff = np.percentile(var_arr, 90)
high_bias = bias_arr >= bias_cutoff
high_var = var_arr  >= var_cutoff
overlap = high_bias & high_var
bias_only = high_bias & (~high_var)
var_only = high_var  & (~high_bias)
plt.scatter(X_test[var_only, 0], X_test[var_only, 1], color="blue")
plt.scatter(X_test[bias_only, 0], X_test[bias_only, 1], color="red")
plt.scatter(X_test[overlap, 0], X_test[overlap, 1], color="orange")
x1_vals = np.array([0, 10])
x2_vals = 2 * x1_vals - 5
plt.plot(x1_vals, x2_vals, color="black", linewidth=2)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show()