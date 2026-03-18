import matplotlib.pyplot as plt

def rysuj_funkcje_kosztu(historia_kosztu):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(historia_kosztu)), historia_kosztu, color='blue')
    plt.xlabel("Iteracja")
    plt.ylabel("Wartość funkcji kosztu")
    plt.title("Funkcja kosztu w kolejnych iteracjach gradientu")
    plt.grid(True)
    plt.show()