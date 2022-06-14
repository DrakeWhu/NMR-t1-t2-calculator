import numpy as np

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