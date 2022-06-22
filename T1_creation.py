"""
This file creates the T1 image
"""

import numpy as np

# We must change this function so it gives the pixel evolution with a variable number of measurements

def pixel_ev(arr, pixel_x,pixel_y): # Function that takes a 3D array of images and the coordinates of a pixel and returns a (2,11) array with the values of the signal of the pixel chosen for TI between (0,1000)
    x = range(0,len(arr)*100,100)
    y = np.array(np.abs(arr[:,pixel_x,pixel_y]))
    pixel_points = np.vstack((x,y))
    return pixel_points

def min_TI_value(arr,x,y): # This function gives the TI value corresponding with the minimal signal
    ev = pixel_ev(arr,x,y)
    min = np.argmin(ev[1,:])
    ti = ev[0,min]
    return ti

# Now we create a function that takes the array of images and the mask, finds for which TI the signal is minimum for each white pixel in the mask
# And then creates a new array where the value on each pixel is the TI where the signal was minimum (which corresponds to T1 for that pixel)

def create_T1_image(mask, image_array):
    
    t1_image = np.empty((60,60))

    for i in range(0,60):
        for j in range(0,60):
            t1_image[i,j]  = np.where(mask[0,i,j] == 1, min_TI_value(image_array,i,j)/np.log(2), 1)

    return t1_image

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
