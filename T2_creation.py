import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.optimize import curve_fit

def pixel_ev_t2(arr,x,y): # Function that takes a 3D 30x60x60 array and the coordinates of a pixel, then returns an array of length 30 containing the signal of that pixel
    ev = np.empty(30)
    for i in range(30):
        ev[i] = arr[i,x,y]
    return ev

def linearize_array(arr): # Function that takes a 1D array of range 30 containing exponential-like data and linearizes it
    lin_ev = np.empty(30)
    lin_ev = np.log(arr)
    return lin_ev

def linear_regression(arr): # Function that makes a linear regression from an 1D array of range 30
    x = range(30)
    A = np.vstack([x,np.ones(len(x))]).T
    m = np.linalg.lstsq(A,arr,rcond=None)
    return m

def create_t2_image(arr,mask): # Takes in a 3D 30x60x60 array, returns a 60x60 image
    analisis = 0
    image = np.empty((60,60))
    for i in range(60):
        for j in range (60):
            ev = pixel_ev_t2(arr,i,j)
            lin_ev = linearize_array(ev)
            m = linear_regression(lin_ev)
            m_b = m[0]
            image[i,j]=np.where(mask[i,j] == 1, np.exp(m_b[0]), 0)
            analisis = analisis + 1
    print(analisis)
    return image

"""
A partir de aquí es del código de Jose
"""

def func(x, m, t2):
    # Función de ajuste
    return m * np.exp(-x / t2)

def getT2(path, nEchos = 60, echoSpacing = 10): # nEchos es el número de ecos, todas 60 menos la del día 1, que tiene 30 (pero con doble tiempo entre ecos). Echo spacing es 20 el primer día
    rawData = scipy.io.loadmat(path)    # Load your rawData here
    imagen = np.abs(rawData['imagen'])
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