import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = np.load("init_dataset_highdim.npz")
X = data["X"]
y = data["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

T = 50

widths = [1,2,3,4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32,35,40,45,50,60,70,80,100,120,150,200,300,400]

train_errors = []
test_errors = []
bias2_list = []
var_list = []
train_mse_list = []

for w in widths:
    tr_list = []
    te_list = []
    mse_list = []
    
    preds = np.zeros((T, len(X_test)))
    
    for t in range(T):
        model = MLPClassifier(
            hidden_layer_sizes=(w,),
            activation="relu",
            learning_rate_init=2e-4,
            alpha=0,
            max_iter=2000,
            early_stopping=False,
            n_iter_no_change=20,
            random_state=t
        )
        
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        y_train_prob = model.predict_proba(X_train)[:,1]
        mse = np.mean((y_train_prob - y_train)**2)
        mse_list.append(mse)

        preds[t] = y_test_pred

        tr_err = np.mean(y_train_pred != y_train)
        te_err = np.mean(y_test_pred != y_test)

        tr_list.append(tr_err)
        te_list.append(te_err)

    train_errors.append(np.mean(tr_list))
    test_errors.append(np.mean(te_list))
    train_mse_list.append(np.mean(mse_list))

    mean_pred = preds.mean(axis=0)

    bias2 = np.mean((mean_pred - y_test)**2)
    variance = np.mean(np.var(preds, axis=0))

    bias2_list.append(bias2)
    var_list.append(variance)


plt.figure(figsize=(8,5))

plt.plot(widths, bias2_list, marker='o', color="#7FBF7B", label="Bias²")
plt.plot(widths, var_list, marker='o', color="#8E7CC3", label="Variance")
plt.plot(widths, test_errors, marker='o', color="#F07C98", label="Test Error")
plt.plot(widths, train_mse_list, marker='o', color="#F5A623", label="Training MSE")

plt.xlabel("Network width")
plt.ylabel("Value")
plt.title("Bias², Variance, Test Error, and Training MSE vs Network Width")

plt.legend()
plt.show()
