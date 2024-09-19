import numpy as np
from time import time

# Program srovnana vybrane funkce modulu numpy (dot, matmul, transpose,
# gradient, trapezoid) s vlastni implementaci a posuzuje je z hlediska casove
# narocnosti. Rozdil je vzdy vypisovan do konzole.

# Pocita skalarni soucin dvou vektoru
def dot(a, b):
    assert len(a) == len(b)  # Vektory musi byt stejne dlouhe
    return sum(a[i] * b[i] for i in range(len(a)))

# Pocita maticovy soucin dvou matic
def matmul(a, b):
    # Dimenze matic
    m = len(a)
    n = len(a[0])
    k = len(b)
    p = len(b[0])

    # Sirka jedne matice se musi shodovat s vyskou druhe
    assert n == k

    c = [[0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            c[i][j] = sum(a[i][k] * b[k][j] for k in range(n))

    return c

# Transponuje matici
def transpose(a):
    return [[a[j][i] for j in range(len(a))] for i in range(len(a[0]))]

# Pocita derivaci (gradient) vektoru
def gradient(a, h=1):
    l = len(a) # delka velkoru
    b = [0] * l

    # Prvni a posledni bod vypocitame doprednou/zpetnou derivaci
    b[0] = (a[1]-a[0])/h
    b[l-1] = (a[l-1]-a[l-2])/h

    # Ostatni body spocitame centralni derivaci
    b[1:l-1] = [(a[i+1]-a[i-1])/(2*h) for i in range(1, l-1)]

    return b

# Pocita urcity integral lichobeznikovym pravidlem
def trapezoid(a, h=1):
    return sum([h*(a[i]+a[i+1])/2 for i in range(len(a)-1)])


def main():
    # Definice objektu, se kterymi budeme ve funkcich pracovat
    SIZE = 200 # Unifikovana velikost objektu
    mat1 = np.random.rand(SIZE, SIZE)
    mat2 = np.random.rand(SIZE, SIZE)
    vec1 = np.random.rand(SIZE)
    vec2 = np.random.rand(SIZE)

    # Seznam paru funkci a jejich argumentu, ktere budeme porovnavat
    FUNCTIONS = [
            (dot, np.dot, [vec1, vec2]),
            (matmul, np.matmul, [mat1, mat2]),
            (transpose, np.transpose, [mat1]),
            (gradient, np.gradient, [vec1]),
            (trapezoid, np.trapezoid, [vec1]),  # Používáme np.trapezoid
            ]

    # Cyklus pres pary funkci a jejich argumenty
    for custom_func, module_func, args in FUNCTIONS:
        func_name = custom_func.__name__

        # Vlastni implementace
        custom_start = time()
        custom_func(*args)
        custom_time = max(time() - custom_start, 1e-9)  # Zajisti, ze cas neni nulovy

        # Numpy
        module_start = time()
        module_func(*args)
        module_time = max(time() - module_start, 1e-9)  # Zajisti, ze cas neni nulovy

        # Vypiseme vysledek mereni
        print(f"{func_name}: NumPy je {custom_time/module_time:.2f}x rychlejsi")

if __name__ == "__main__":
    main()
