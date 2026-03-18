import numpy as np

def regresjaLiniowa (X, theta):
    h = np.dot(X, theta)
    return h

def funkcjaKosztu(X,Y,theta):
    m = X.shape[0]
    h = regresjaLiniowa(X,theta)
    J = (1/(2*m))*np.sum((h-Y)**2)
    return J

def gradientDescent(X, Y, theta, alpha, num_iters):
    m = X.shape[0]
    J_history = []
    X_T = np.transpose(X)


    for i in range(num_iters):
        h = regresjaLiniowa(X, theta)

        theta = theta - alpha * (1 / m)* X_T.dot(h - Y)
        J = funkcjaKosztu(X, Y, theta)
        J_history.append(J)

    return theta, J_history

def predykcja(X_test, theta):
    return np.dot(X_test, theta)