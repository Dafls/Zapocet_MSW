import numpy as np
from time import time
from scipy.linalg import solve # Prima metoda
import matplotlib.pyplot as plt

# Program zmeri casovou narocnost reseni soustav rovnic o velikostech
# MATRIX_SIZES pomoci prime a iteracni metody. Toto provede tolikrat,
# kolik je SAMPLE_SIZE, a nasledne z namerenych hodnot spocita prumer. Prumerne
# hodnoty jsou pote zaneseny do grafu.

# Iteracni metoda (z vyukovych materialu)
def gauss_seidel(A, b, niteraci, x0):
    x = x0
    U = np.triu(A, k = 1)
    Lstar = np.tril(A, k = 0)
    T = np.matmul(-np.linalg.inv(Lstar), U)
    C = np.matmul(np.linalg.inv(Lstar), b)
    for i in range(niteraci):
        x = np.matmul(T, x) + C
    return x

# Generuje resitelne soustavy rovnic
def generate_equations(size):
    while True:
        A = np.random.randint(-10, 10, (size, size))
        np.fill_diagonal(A, np.sum(np.abs(A), axis=1) + 1) # Resi singular matrix error
        if np.linalg.matrix_rank(A) == size:
            break
    x = np.random.randint(-10, 10, size)
    b = np.dot(A, x)
    return A, b

# Meri trvani prime metody
def measure_direct_method(A,b):
    start = time()
    solve(A,b)
    return time() - start

# Meri trvani iteracni metody
def measure_iterative_method(A,b):
    start = time()
    gauss_seidel(A, b, 10, np.ones(len(A)))
    return time() - start
    return stop - start

def main():
    MATRIX_SIZES = [5,10,20,50,100,200,500] # Matice jakych velikosti otestovat
    SAMPLE_SIZE = 10 # Z kolika vypoctu se dela prumer

    direct_times_mean = np.zeros((len(MATRIX_SIZES))) 
    iterative_times_mean = np.zeros((len(MATRIX_SIZES)))

    # Cyklus pro prumerovani
    for _ in range(SAMPLE_SIZE):
        # Vygenerujeme soustavy jako list dvojic (A, b)
        equations = [generate_equations(size) for size in MATRIX_SIZES]

        # Zmerime trvani metod na kazde ze soustav
        direct_times = [measure_direct_method(A,b) for A,b in equations]
        iterative_times = [measure_iterative_method(A,b) for A,b in equations]

        # Pricteme namerene hodnoty
        direct_times_mean += direct_times
        iterative_times_mean += iterative_times

    # Prumer ziskame jako suma namerenych hodnot delela poctem vzorku
    direct_times_mean = direct_times_mean / SAMPLE_SIZE
    iterative_times_mean = iterative_times_mean / SAMPLE_SIZE

    # Vykreslime graf
    plt.figure(figsize=(10, 6))
    plt.plot(MATRIX_SIZES, direct_times_mean, label="Přímá metoda", marker="o")
    plt.plot(MATRIX_SIZES, iterative_times_mean, label="Iterační metoda", marker="o")
    plt.xlabel("Velikost matice")
    plt.ylabel("Čas")
    plt.title("Porovnání časů pro přímé a iterační metody")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
