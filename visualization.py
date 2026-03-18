import matplotlib.pyplot as plt
import numpy as np

def rysuj_funkcje_kosztu(historia_kosztu):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(historia_kosztu)), historia_kosztu, color='red')
    plt.xlabel("Iteracja")
    plt.ylabel("Wartość funkcji kosztu")
    plt.title("Funkcja kosztu w kolejnych iteracjach gradientu")
    plt.grid(True)
    plt.savefig("wykres_kosztu.png")
    plt.show()

def oblicz_i_wypisz_wyniki(Y_test, Y_pred, theta):
    ss_res = np.sum((Y_test - Y_pred)**2)
    ss_tot = np.sum((Y_test - np.mean(Y_test))**2)
    r2 = 1 - ss_res/ss_tot
    print(r2)
    print(theta)