import numpy as np
import time

# Program projde funkce z FUNCTIONS a nalezne koren pomoci bisekce a Newtonovy
# metody. Na konzoli pote pro kazdou z metod vypise nalezeny koren, dobu behu a
# presnost.

# Polynomialni funkce
def poly_func(x):
    return x**3 - 2*x - 5

# Exponencialni funkce
def exp_func(x):
    return np.exp(x) - 3

# Trigonometricka funkce
def trigon_func(x):
    return np.sin(x) - x / 10

# Derivace funkci pro Newtonovu metodu
def poly_deriv(x):
    return 3*x**2 - 2

def exp_deriv(x):
    return np.exp(x)

def trigon_deriv(x):
    return np.cos(x) - 1 / 10

# Metoda puleni intervalu (uzavrena, z internetu)
def bisection(f, a, b, tol=1e-6): 
    if np.sign(f(a)) == np.sign(f(b)):
        raise Exception("Funkce musí mít na koncových bodech odlišné znaménko.")
    m = (a + b)/2
    if np.abs(f(m)) < tol:
        return m
    elif np.sign(f(a)) == np.sign(f(m)):
        return bisection(f, m, b, tol)
    elif np.sign(f(b)) == np.sign(f(m)):
        return bisection(f, a, m, tol)

# Newtonova metoda (otevrena, z internetu)
def newton(f, df, x0, tol=1e-6):
    if abs(f(x0)) < tol:
        return x0
    else:
        return newton(f, df, x0 - f(x0)/df(x0), tol)

def main():
    FUNCTIONS = [poly_func, exp_func, trigon_func]
    DERIVATIVES = [poly_deriv, exp_deriv, trigon_deriv]
    INITIAL_GUESSES = [1, 1, 1]  # Pocatecni odhady pro Newtonovu metodu

    for func, deriv, guess in zip(FUNCTIONS, DERIVATIVES, INITIAL_GUESSES):
        print(f"{func.__name__}():")
        
        print("  bisection():")
        start = time.time()
        root_bisection = bisection(func, -10, 10)
        time_bisection = time.time() - start
        print(f"    koren: {root_bisection:.6f}")
        print(f"    cas: {time_bisection:.6f}")
        print(f"    presnost: {abs(func(root_bisection)):.6e}")
        
        print("  newton():")
        start = time.time()
        root_newton = newton(func, deriv, guess)
        time_newton = time.time() - start
        print(f"    koren: {root_newton:.6f}")
        print(f"    cas: {time_newton:.6f}")
        print(f"    presnost: {abs(func(root_newton)):.6e}")

if __name__ == "__main__":
    main()
