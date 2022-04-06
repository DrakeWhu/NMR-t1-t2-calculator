import scipy.signal as sig
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from pathlib import Path

data_path = Path(r"C:/Users/juanr/Desktop/Universidad/Master/Practicas/programa calculo t1t2/NMR-t1-t2-calculator/rawdata/")

onlyfiles = [str(pp) for pp in data_path.glob("**/*.mat")]

def get_slice_number(file_path): # This function gives the number of slices of a 3D image
    rawData = scipy.io.loadmat(file_path)
    slices = rawData['nPoints'][0][2]
    return slices


def import_2Dimage(file_path,i): # This function extracts only the 2D image of a chosen slice indexed by i
    rawData = scipy.io.loadmat(file_path)
    image = rawData['image3D']
    image2d = image[i,:,:]
    return image2d

def create_image_array(): # This function creates the array that contains the images
    image_paths_array = []
    image_array = []
    for i in range(len(onlyfiles)):
        path = onlyfiles[i]
        image_paths_array.insert(i,path)
        image2D = import_2Dimage(image_paths_array[i],round(get_slice_number(onlyfiles[i]) * 0.5))
        image_array.insert(i, image2D)
        i = i + 1
    return image_array

image_array = create_image_array()

print(len(image_array))

def separate_t1_t2(): # This function separates the images used to calculate T1 and T2
    t1_images = []
    t2_images = []
    for i in range(len(image_array)):
        image = image_array[i]
        height = len(image[0,:])
        width = len(image[:,0])
        if width > 5*height: # Here we choose a naïve way of separating: the width must be 5 times the height or bigger. We do this cause mse generates really wide images
            t2_images.insert(i, image)
        else:
            t1_images.insert(i, image)
        i = i+1
    return t1_images, t2_images

# We create the separated arrays

separated_arrays = separate_t1_t2()
t1_list = separated_arrays[0]
t2_list = separated_arrays[1]

# We turn them into numpy arrays

t1 = np.asarray(t1_list)
t2 = np.asarray(t2_list)

t1_orange_seedless = t1[0:11]
t1_orange_seeds = t1[11:22]
t1_banana_d1 = t1[22:35]
t1_banana_bag_d1 = t1[35:48]

# print(t1_seeds[0,:,:]) # [image number, height, width]

def pixel_ev(arr, pixel_x,pixel_y): # Function that takes a 3D array of images and the coordinates of a pixel and returns a (2,11) array with the values of the signal of the pixel chosen for TI between (0,1000)
    x = range(0,1300,100)
    y = np.array(np.abs(arr[:,pixel_x,pixel_y]))
    pixel_points = np.vstack((x,y))
    #print(pixel_points)
    #plt.scatter(pixel_points[0,:],pixel_points[1,:])
    #plt.show()
    return pixel_points
    
# We must change this function so it gives the pixel evolution with a variable number of measurements

def min_TI_value(arr,x,y): # This function gives the TI value corresponding with the minimal signal
    ev = pixel_ev(arr,x,y)
    min = np.argmin(ev[1,:])
    ti = ev[0,min]
    return ti

orange_seeds_mask = np.where(np.abs(t1_orange_seeds) < 0.0005,0,1)
orange_seedless_mask = np.where(np.abs(t1_orange_seedless) < 0.0005,0,1)
banana_mask = np.where(np.abs(t1_banana_d1) < 0.002,0,1)
banana_bag_mask = np.where(np.abs(t1_banana_bag_d1) < 0.002,0,1)

# Now we create a function that takes the array of images and the mask, finds for which TI the signal is minimum for each white pixel in the mask
# And then creates a new array where the value on each pixel is the TI where the signal was minimum (which corresponds to T1 for that pixel)

def create_T1_image(mask, image_array):
    analisis = 0
    t1_image = np.empty((60,60))

    """
    plt.figure(1)
    plt.imshow(mask[0,:,:], cmap='gray')
    plt.show()
    """

    for i in range(0,60):
        for j in range(0,60):
            t1_image[i,j]  = np.where(mask[0,i,j] == 1, min_TI_value(image_array,i,j)/np.log(2), 1)
            analisis = analisis + 1
    
    print(analisis) # This variable keeps track of how many loops have been done (useful for debugging)

    return t1_image

#t1_seeds_image = create_T1_image(orange_seeds_mask,np.abs(t1_orange_seeds))
#t1_seedless_image = create_T1_image(orange_seedless_mask,np.abs(t1_orange_seedless))
t1_banana_image = create_T1_image(banana_mask,np.abs(t1_banana_d1))
t1_banana_bag_image = create_T1_image(banana_bag_mask,np.abs(t1_banana_bag_d1))

# With this next three lines we choose what image we want to see (mostly for testing)

# Now we are going to work with the mse image in order to extract the T2
#t2_seedless = t2[0] # This mse is on the wrong slice, we should use a more representative slice
#t2_seeds = t2[1]

def create_t2_array(arr): # Function that we pass a 60x1800 2D array and creates a 30x60x60 3D array
    array_3D =  np.empty((30,60,60))
    for i in range(30):
        array_3D[i,:,:] = np.abs(arr[0:60,60*i:60*i+60])
    return array_3D

#t2_seeds_3D_array = create_t2_array(t2_seeds)
#t2_seedless_3D_array = create_t2_array(t2_seedless)

#t2_seeds_mask = np.where(np.abs(t2_seeds_3D_array[0,:,:])<0.0005,0,1)
#t2_seedless_mask = np.where(np.abs(t2_seedless_3D_array[0,:,:])<0.0005,0,1)

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

m = linear_regression(range(30))
print(m[0])

def create_t2_image(arr,mask): # Takes in a 3D 30x60x60 array, returns a 60x60 image
    image = np.empty((60,60))
    for i in range(60):
        for j in range (60):
            ev = pixel_ev_t2(arr,i,j)
            lin_ev = linearize_array(ev)
            m = linear_regression(lin_ev)
            m_b = m[0]
            image[i,j]=np.where(mask[i,j] == 1, np.exp(m_b[0]), 0)
    return image

#t2_seeds_image = create_t2_image(t2_seeds_3D_array,t2_seeds_mask)
#t2_seedless_image = create_t2_image(t2_seedless_3D_array,t2_seedless_mask)

plt.figure(1)
plt.imshow(np.abs(t1_banana_image), cmap='inferno')
plt.colorbar()
#plt.clim(0.9,1.02)
plt.show()
