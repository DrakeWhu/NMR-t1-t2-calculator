import scipy.signal as sig
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

data_path = "C:/Users/juanr/Desktop/Universidad/Master/Practicas/programa calculo t1t2/NMR-t1-t2-calculator/data"

onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

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

# With this next three lines we choose what image we want to see (mostly for testing)

plt.figure(1)
plt.imshow(np.abs(image_array[0]), cmap='gray')
plt.show()
