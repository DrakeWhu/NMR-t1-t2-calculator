import scipy.signal as sig
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from pathlib import Path

t1_data_path = Path(r"C:/Users/juanr/Desktop/Universidad/Master/Practicas/programa calculo t1t2/NMR-t1-t2-calculator/rawdata/")
t2_data_path = Path(r"C:/Users/juanr/Desktop/Universidad/Master/Practicas/programa calculo t1t2/NMR-t1-t2-calculator/t2_raw_data/")

t1_onlyfiles = [str(pp) for pp in t1_data_path.glob("**/*.mat")]
t2_onlyfiles = [str(pp) for pp in t2_data_path.glob("**/*.mat")]

def get_slice_number(file_path,str): # This function gives the number of slices of a 3D image
    rawData = scipy.io.loadmat(file_path)
    image = rawData[str]
    height = len(image[0,:])
    width = len(image[:,0])

    if width > 5*height:
        slices = rawData['nPoints'][0][2]
        return slices
    else:
        return 1


def import_2Dimage(file_path,i,str): # This function extracts only the 2D image of a chosen slice indexed by i
    rawData = scipy.io.loadmat(file_path)
    image = rawData[str]
    image2d = image[i,:,:]
    return image2d

def create_image_array(onlyfiles,str): # This function creates an array that contains the images
    image_paths_array = []
    image_array = []
    for i in range(len(onlyfiles)):
        path = onlyfiles[i]
        image_paths_array.insert(i,path)
        image2D = import_2Dimage(image_paths_array[i],round(get_slice_number(onlyfiles[i],str) * 0.5),str)
        image_array.insert(i, image2D)
        i = i + 1
    return image_array

t1_image_array = create_image_array(t1_onlyfiles,'image3D')
t2_image_array = create_image_array(t2_onlyfiles,'imagen')


def separate_t1_t2(): # This function separates the images used to calculate T1 and T2
    t1_images = []
    t2_images = []
    for i in range(len(t1_image_array)):
        image = t1_image_array[i]
        height = len(image[:,0])
        width = len(image[0,:])
        if width == 1800: # Here we choose a naïve way of separating: the width must be 5 times the height or bigger. We do this cause mse generates really wide images
            t2_images.insert(i, image)
        else:
            t1_images.insert(i, image)
        i = i+1
    return t1_images, t2_images


# We create the separated arrays

separated_arrays = separate_t1_t2()
t1_list = separated_arrays[0]
t2_oranges_list = separated_arrays[1]

print(t2_oranges_list)

# We turn them into numpy arrays

t1 = np.asarray(t1_list)
t2 = np.asarray(t2_image_array)
t2_oranges = np.asarray(t2_oranges_list)

print(t2)

t1_orange_seedless = t1[0:11]
t1_orange_seeds = t1[11:22]
t1_banana_d1 = t1[22:35]
t1_banana_bag_d1 = t1[35:48]
t1_banana_d2 = t1[48:61]
t1_banana_bag_d2 = t1[61:74]
t1_banana_d3 = t1[74:87]
t1_banana_bag_d3 = t1[87:100]
t1_banana_d4 = t1[100:113]
t1_banana_bag_d4 = t1[113:126]

# print(t1_seeds[0,:,:]) # [image number, height, width]

def pixel_ev(arr, pixel_x,pixel_y): # Function that takes a 3D array of images and the coordinates of a pixel and returns a (2,11) array with the values of the signal of the pixel chosen for TI between (0,1000)
    x = range(0,len(arr)*100,100)
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

plt.figure(1)
plt.imshow(np.abs(t1_orange_seeds[0,:,:]), cmap='gray')
plt.show()

# Now we create a function that takes the array of images and the mask, finds for which TI the signal is minimum for each white pixel in the mask
# And then creates a new array where the value on each pixel is the TI where the signal was minimum (which corresponds to T1 for that pixel)

def create_T1_image(mask, image_array):
    t1_image = np.empty((60,60))

    """
    plt.figure(1)
    plt.imshow(mask[0,:,:], cmap='gray')
    plt.show()
    """

    for i in range(0,60):
        for j in range(0,60):
            t1_image[i,j]  = np.where(mask[0,i,j] == 1, min_TI_value(image_array,i,j)/np.log(2), 1)

    return t1_image

t1_seeds_image = create_T1_image(orange_seeds_mask,np.abs(t1_orange_seeds))
t1_seedless_image = create_T1_image(orange_seedless_mask,np.abs(t1_orange_seedless))
t1_banana_image = create_T1_image(banana_mask,np.abs(t1_banana_d1))
t1_banana_bag_image = create_T1_image(banana_bag_mask,np.abs(t1_banana_bag_d1))

# With this next three lines we choose what image we want to see (mostly for testing)

# Now we are going to work with the mse image in order to extract the T2
t2_seedless = t2_oranges[0]
t2_banana_d1 = t2[0]

def create_t2_array(arr): # Function that we pass a 60x1800 2D array and creates a 30x60x60 3D array
    array_3D =  np.empty((30,60,60))
    for i in range(30):
        array_3D[i,:,:] = np.abs(arr[0:60,60*i:60*i+60])
    return array_3D

t2_seedless_3D_array = create_t2_array(t2_seedless)
t2_banana_3D_array_d1 = create_t2_array(t2_banana_d1)
 
t2_seedless_mask = np.where(np.abs(t2_seedless_3D_array[0,:,:])<0.0005,0,1)

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

t2_seedless_image = create_t2_image(t2_seedless_3D_array,t2_seedless_mask)
#t2_banana_image_d1 = create_t2_image(t2_banana_3D_array_d1,banana_mask)

plt.figure(1)
plt.imshow(np.abs(t1_seedless_image), cmap='inferno')
plt.colorbar()
#plt.clim(0.9,1)
plt.show()
