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

def getT1(imagen, nImages, tVector, nPoints, pixel):
    imagen2d = np.squeeze(imagen[int(nPoints[2]/2), :, :]) # Tomo el segundo slice para los cálculos

    # Aquí paso las diferentes imagenes una matriz de 3 dimensioens, la primera dimensión da cuenta del echo
    imagen3d = np.zeros((nImages, nPoints[1], nPoints[0]))
    for ima in range(nImages):
        imagen3d[ima, :, :] = imagen2d[:, nPoints[0]*ima:nPoints[0]*ima+nPoints[0]]

    sVector = imagen3d[:, pixel[1], pixel[0]]
    idx = np.argmin(sVector)
    inversionTime = tVector[idx]
    T1 = inversionTime/np.log(2)

    return T1

def createImageArray(name, n):
    images = np.zeros((4, 60, 60))
    for ima in range(n):
        fullName = name+' ('+str(ima+1)+').mat'
        rawData = scipy.io.loadmat(fullName)
        images = np.concatenate((images, np.abs(rawData['image3D'])), axis=2)

    return images[:, :, 60::]




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    calcular = 'T2' # aquí puedes poner T1 o T2 según lo que quieras, tendrás que modificar la sección correspondiente

    if calcular == 'T2':
        rawData = scipy.io.loadmat('./t2_banana/MSE.2022.04.07.17.03.09.mat')  # Load your rawData here
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

    elif calcular == 'T1':
        rawData = scipy.io.loadmat('./t1Outside-2022.04.01/Old_RARE.2022.04.01 (1).mat')
        nPoints = np.array(rawData['nPoints'][0])
        nImages = 13
        tVector = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000])
        name = './t1Outside-2022.04.01/Old_RARE.2022.04.01' # Directory and name until the day
        imagen = createImageArray(name, nImages)

        # Get mask
        mask = np.squeeze(imagen[int(nPoints[2] / 2), :, :])
        mask = mask / np.max(mask[:])
        mask[mask < 0.15] = 0
        mask[mask >= 0.15] = 1

        # Get t2 for all pixels (I use ms as time units)
        t1Map = np.zeros((nPoints[1], nPoints[0]))
        for x in range(nPoints[1]):
            for y in range(nPoints[0]):
                if mask[y, x]==1:
                    t1Map[y, x] = getT1(imagen, nImages, tVector, nPoints, [x, y])
        t1Map[t1Map>2e3] = 0    # Just in case there is a huge T2 value due to fitting error

        plt.figure(1)
        plt.imshow(t1Map)
        plt.colorbar()
        plt.show()



