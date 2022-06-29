"""
This file takes the image data from the .mat format and extracts the different measurements as an array
"""
import scipy.io
import numpy as np


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

def create_t2_array(arr): # Function that we pass a 60x1800 2D array and creates a 30x60x60 3D array
    array_3D =  np.empty((30,60,60))
    for i in range(30):
        array_3D[i,:,:] = np.abs(arr[0:60,60*i:60*i+60])
    return array_3D

def createImageArray(path, name, n):
    images = np.zeros((10, 60, 60))
    for ima in range(n):
        fullName = path+' ('+str(ima+1)+')'+name+' ('+str(ima+1)+').mat'
        rawData = scipy.io.loadmat(fullName)
        images = np.concatenate((images, np.abs(rawData['image3D'])), axis=2)

    return images[:, :, 60::]