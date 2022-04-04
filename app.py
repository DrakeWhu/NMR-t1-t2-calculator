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

print(t2[:,0,0]) # [image number, width, heigh]

# With this next three lines we choose what image we want to see (mostly for testing)

plt.figure(1)
plt.imshow(np.abs(t1[0]), cmap='gray')
plt.show()
