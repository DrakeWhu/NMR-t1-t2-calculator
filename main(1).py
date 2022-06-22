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

def getT2(imagen, nEchos, echoSpacing, nPoints, pixel):
    imagen2d = np.squeeze(imagen[int(nPoints[2]/2), :, :]) # Tomo el segundo slice para los cálculos

    # Aquí paso las diferentes imagenes una matriz de 3 dimensioens, la primera dimensión da cuenta del echo
    imagen3d = np.zeros((nEchos, nPoints[1], nPoints[0]))
    for echo in range(nEchos):
        imagen3d[echo, :, :] = imagen2d[:, nPoints[0]*echo:nPoints[0]*echo+nPoints[0]]

    tVector = (np.arange(0, nEchos)+1)*echoSpacing
    sVector = imagen3d[:, pixel[1], pixel[0]]
    try:
        fitData, xxx = curve_fit(func, tVector, sVector)    # Ajuste a la función definida en func
        T2 = fitData[1]  # ms, T2 obtenido en el ajuste
    except:
        T2 = 0

    return T2


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rawData = scipy.io.loadmat('./t2Orange/MSE.2022.03.30.09.52.24.mat')  # Load your rawData here
    imagen = np.abs(rawData['imagen'])
    nEchos = int(rawData['etl'][0][0])  # number of echoes
    echoSpacing = rawData['echoSpacing'][0][0] * 1e3,  # ms
    nPoints = np.array(rawData['nPoints'][0])
    # pixel = [32, 15]

    # Get mask
    mask = np.squeeze(imagen[int(nPoints[2]/2), :, 0:nPoints[0]])
    mask = mask/np.max(mask[:])
    mask[mask<0.08] = 0
    mask[mask>=0.08] = 1

    # Get t2 for all pixels (I use ms as time units)
    t2Map = np.zeros((nPoints[1], nPoints[0]))
    for x in range(nPoints[1]):
        for y in range(nPoints[0]):
            if mask[y, x]==1:
                t2Map[y, x] = getT2(imagen, nEchos, echoSpacing, nPoints, [x, y])
    t2Map[t2Map>2e3] = 0    # Just in case there is a huge T2 value due to fitting error

    plt.figure(1)
    plt.imshow(t2Map, vmin=200, vmax=1200)
    plt.colorbar()
    plt.show()

