import scipy.signal as sig
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

data_path = "C:/Users/juanr/Desktop/Universidad/Master/Practicas/programa calculo t1t2/NMR-t1-t2-calculator/data"

onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))] # We should find a way to read a folder of folders and dump each folder in an array (now we can only read a folder of files)

def path_file_concatenate_strings(i): # This function creates the path string
    path = "./data/"
    filename = onlyfiles[i]
    str = path + filename
    return str

def import_2Dimage(file_path,i): # This function extracts only the 2D image of a chosen slice indexed by i
    rawData = scipy.io.loadmat(file_path)
    image = rawData['image3D']
    image2d = image[i,:,:]
    return image2d

def create_image_array(): # This function creates the array that contains the images
    i = 0
    image_paths_array = []
    image_array = []
    while i <= len(onlyfiles) - 1:
        path = path_file_concatenate_strings(i)
        image_paths_array.insert(i,path)
        image2D = import_2Dimage(image_paths_array[i],5) # Here we choose 5 because we are working with 10 slices, but the ideal would be to use the midpoint of the array
        image_array.insert(i, image2D)
        i = i + 1
    return image_array

image_array = create_image_array()

def separate_t1_t2(): # This function separates the images used to calculate T1 and T2
    i = 0
    t1_images = []
    t2_images = []
    while i <= len(image_array) - 1:
        image = image_array[i]
        height = len(image)
        width = len(image[i])
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

t1_seedless = t1[0:11]
t1_seeds = t1[11:22]

# print(t1_seeds[0,:,:]) # [image number, height, width]

def pixel_ev(arr, pixel_x,pixel_y): # Function that takes a 3D array of images and the coordinates of a pixel and returns a (2,11) array with the values of the signal of the pixel chosen for TI between (0,1000)
    x = np.arange(0,1100,100)
    y = np.array(np.abs(arr[:,pixel_x,pixel_y]))
    pixel_points = np.vstack((x,y))
    #print(pixel_points)
    #plt.scatter(pixel_points[0,:],pixel_points[1,:])
    #plt.show()
    return pixel_points

def min_TI_value(arr,x,y): # This function gives the TI value corresponding with the minimal signal
    ev = pixel_ev(arr,x,y)
    min = np.argmin(ev[1,:])
    ti = ev[0,min]
    return ti

seeds_mask = np.where(np.abs(t1_seeds) < 0.0005,0,1)
seedless_mask = np.where(np.abs(t1_seedless) < 0.0005,0,1)

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
            t1_image[i,j]  = np.where(mask[0,i,j] == 1, min_TI_value(image_array,i,j), 1)
            analisis = analisis + 1
    
    print(analisis) # This variable keeps track of how many loops have been done (useful for debugging)

    return t1_image

t1_seeds_image = create_T1_image(seeds_mask,np.abs(t1_seeds))
t1_seedless_image = create_T1_image(seedless_mask,np.abs(t1_seedless))

# With this next three lines we choose what image we want to see (mostly for testing)

plt.figure(1)
plt.imshow(np.abs(t1_seeds_image), cmap='inferno')
plt.colorbar()
plt.show()
