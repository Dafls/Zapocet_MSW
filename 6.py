from PIL import ImageGrab
import numpy as np

# Program poridi snimek obrazovky a secte hodnoty jednotlivych pixelu modulo
# MAX. Potrebnou entropii ziskava, pokud uzivatel zarizeni aktivne pouziva --
# ma otevrena ruzna okna apod.

def main():
    MAX = 2**16 # Generujeme cislo v intervalu <0; MAX)

    img = ImageGrab.grab()
    arr = np.array(img)
    num = np.sum(arr) % MAX
    print(num)

if __name__ == "__main__":
    main()

