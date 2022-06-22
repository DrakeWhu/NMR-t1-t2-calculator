# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, m, t2):
    # Función de ajuste
    return m * np.exp(-x / t2)

def getT2():
    rawData = scipy.io.loadmat('./t2_raw_data/2022.04.06.15.18.38/MSE.2022.04.06.15.18.38.mat')    # Load your rawData here
    imagen = np.abs(rawData['imagen'])
    nEchos = 60 # número de ecos, todas 60 menos la del día 1, que tiene 30 (pero con doble tiempo entre ecos)
    echoSpacing = 10, # ms, 20 ms si es la imagen del día 1
    nPoints = [60, 60, 4] # Las imágenes del día 4 y 5 tienen un solo slice, luego no son válidas para el análisis
    imagen2d = np.squeeze(imagen[2, :, :]) # Tomo el segundo slice para los cálculos

    # Aquí paso las diferentes imagenes una matriz de 3 dimensioens, la primera dimensión da cuenta del echo
    imagen3d = np.zeros((nEchos, nPoints[1], nPoints[0]))
    for echo in range(nEchos):
        imagen3d[echo, :, :] = imagen2d[:, nPoints[0]*echo:nPoints[0]*echo+nPoints[0]]

    # tVector = np.linspace(echoSpacing, nEchos*echoSpacing, nEchos) # ms, Vector de tiempos de echo
    tVector = (np.arange(0,nEchos)+1)*echoSpacing
    sVector = imagen3d[:, 30, 33]   # Uso el pixel (35, 30) para este rawData. Comprobar el pixel en la figura 2D para cada rawData
    fitData, xxx = curve_fit(func, tVector, sVector)    # Ajuste a la función definida en func
    T2 = fitData[1] # ms, T2 obtenido en el ajuste

    # Mapa 2D del primer echo. Usar para ver el pixel en que se quiere obtener T2
    plt.figure(1)
    plt.imshow(np.squeeze(imagen3d[10, :, :]), cmap='gray') # Primer echo, podrías explorar varios para ver

    # Plot de datos experimentales y curva de ajuste
    plt.figure(2)
    plt.plot(tVector, sVector, tVector, func(tVector, *fitData))
    plt.ylabel('Pixel signal amplitude (a.u.)')
    plt.xlabel('Echo time (ms)')
    plt.title('T2 = ' + str(np.around(T2)) + 'ms')

    plt.show()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    getT2()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
