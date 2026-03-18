import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import gradientDescent
import json
from sklearn.model_selection import train_test_split
from visualization i
std_values = X_train.std()

X_train_norm = (X_train-mean_values)/std_values
X_test_norm = (X_test-mean_values)/std_values

X_train = X_train_norm.values
X_test = X_test_norm.values
Y_train = Y_train.values.reshape(-1,1)
Y_test = Y_test.values.reshape(-1,1)

m = X_train.shape[0]
X_train = np.c_[np.ones(m),X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

theta = np.zeros((X_train.shape[1], 1))

alpha = dane["alpha"]
num_iters = dane["num_iters"]

theta, cost = gradientDescent(X_train, Y_train, theta, alpha, num_iters)
Y_pred = np.dot(X_test, theta)

plt.scatter(Y_test, Y_pred)
max_val = max(Y_test.max(), Y_pred.max())
plt.plot([0, max_val], [0, max_val], color="black")
plt.show()



rysuj_funkcje_kosztu(cost)

oblicz_i_wypisz_wyniki(Y_test, Y_pred, theta)