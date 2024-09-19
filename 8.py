import numpy as np
import matplotlib.pyplot as plt

# Program pocita derivaci funkce f() analytickym zpusobem, se statickym krokem
# a s adaptabilnim krokem. Nasledne vypise chybu metod s ruznou volnou kroku
# oproti analytickemu reseni a vse vykresli pomoci pyplot.

# Funkce, se kterou budeme pracovat
def f(x):
    return np.sin(x)

# Analyticka derivace funkce vyse
def f_diff_analytical(x):
    return np.cos(x)

# Vraci doprednou derivaci v bode x
def diff(f, x, h=0.1):
    return (f(x + h) - f(x)) / h

# Vraci derivaci s adaptibilnim krokem v bode x
def adaptive_diff(f, x, h0=0.1, tol=1e-3):
    h = h0
    while True:
        d1 = diff(f, x, h)
        d2 = diff(f, x, h / 2)
        err = abs(d1-d2)
        if err < tol:
            return d2
        h /= 2

# Pocita chybu jako soucet rozptylu
def total_error(y1, y2):
    return np.sum((y1-y2)**2)
    
def main():
    # Definicni obor
    x = np.linspace(0, 1, 100)

    # Ziskame derivace ruznymi metodami
    y_analytical = f_diff_analytical(x)
    y_forward = [diff(f,x[idx]) for idx in range(len(x))]
    y_adaptive = [adaptive_diff(f,x[idx]) for idx in range(len(x))]

    # Vypiseme chybu vzhledem k analytickemu reseni
    print("chyba:")
    print(f"  staticky krok: {total_error(y_analytical, y_forward):.6e}")
    print(f"  adaptabilni krok: {total_error(y_analytical, y_adaptive):.6e}")

    # Vykreslime do grafu
    plt.figure(figsize=(8, 6))
    plt.plot(x, y_analytical, "-", color="orange", label="Analytické řešení")
    plt.plot(x, y_forward, "-", color="lightblue", label="Statický krok")
    plt.plot(x, y_adaptive, ":", color="darkred", label="Adaptabilní krok")
    plt.title(f"Srovnání metod derivace")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

