import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import gradientDescent
import json
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:\\Moje\\Projects\\PyCharm projects\\PythonProject\\data\\train.csv")
with open("plik.json", "r", encoding="utf-8") as plik:
    dane = json.load(plik)

data["sex"]=data["sex"].map({"male":1,"female":0})
data["smoker"] = data["smoker"].map({"yes":1,"no":0})

X = data[["age","sex","bmi","children","smoker"]]
Y = data[["charges"]]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state = 42)

mean_values = X_train.mean()
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

ss_res = np.sum((Y_test - Y_pred)**2)
ss_tot = np.sum((Y_test - np.mean(Y_test))**2)
r2 = 1 - ss_res/ss_tot
print(r2)
print(theta)

from visualization import rysuj_funkcje_kosztu
rysuj_funkcje_kosztu(cost)