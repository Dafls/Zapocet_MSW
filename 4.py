import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline

# Program prochazi funkce obsazene ve FUNCTIONS, na ktere aplikuje sum a filtr
# tak, aby napodobil realne mereni. Dale na tyto hodnoty aplikuje ruzne metody
# interpolace/aproximace z METHODS, pricemz vysledek je vzdy vykreslen pomoci
# pyplot.

# Linearni funkce
def linear_func(x):
    return x

# Logaritmicka funkce
def logarithmic_func(x):
    return np.log(x)

# Trigonometricka funkce
def trigonometric_func(x):
    return np.sin(x)

# Pridava sum k datum
def add_noise(y, sigma=0.05):
    noise = np.random.normal(0, sigma, y.shape)
    return y + noise

# Vybere nahodnou podmnozinu hodnot
def select_subset(x, y, ratio=0.7):
    mask = np.random.rand(len(x)) < ratio
    return x[mask], y[mask]

# Provede linearni interpolaci (vraci funkci)
def linear_interp(x, y):
    return interp1d(x, y, fill_value="extrapolate")

# Provede kubickou interpolaci (vraci funkci)
def cubic_spline_interp(x, y):
    return CubicSpline(x, y, extrapolate=True)

# Provede polynomialni approximaci (vraci funkci)
def polynomial_approx(x, y):
    return np.poly1d(np.polyfit(x, y, 7))

FUNCTIONS = [linear_func, logarithmic_func, trigonometric_func]
METHODS = [linear_interp, cubic_spline_interp, polynomial_approx]

def main():

    # Definicni obor
    x = np.linspace(0.1, 8, 100)

    # Cyklus pres funkce
    for func in FUNCTIONS:
        y = func(x)
        x_sparse, y_sparse = select_subset(x, add_noise(y)) # Sum a filtr -- imituje nepresnosti mereni

        # Cyklus pres metody
        for method in METHODS:
            y_new = method(x_sparse, y_sparse)(x) # Hodnoty ziskane interpolaci/aproximaci

            # Vykreslit do grafu
            plt.figure(figsize=(8, 6))
            plt.plot(x_sparse, y_sparse, "o", label="Naměřené hodnoty")
            plt.plot(x, y_new, "-", label="Proložená funkce")
            plt.title(f"Proložení {func.__name__}() metodou {method.__name__}()")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.grid(True)
            plt.show()

if __name__ == "__main__":
    main()
