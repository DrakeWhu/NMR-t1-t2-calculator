"""
This file creates the evolution of T1 for a given array of T1 images
"""

import numpy as np

def average(image): # This function takes a 60x60 image and returns the average pixel value of a 5x5 group of pixels

    sum = 0
    total_pixels = 0

    for i in range(30,36):
        for j in range(25,31):
            if image[i, j] != 0:
                total_pixels += 1
                sum += abs(image[i,j])
    
    t1_average = sum/total_pixels
    return t1_average



def evolution(array): # This function takes an array of 60x60 images and makes an array of the averages of a voxel of pixels in the meat part of the image

    average_t1_array = np.empty(4)

    for i in range(4):
        mean = average(abs(array[i,:,:]))
        np.insert(average_t1_array, i, mean)
    
    return average_t1_array

def linear_regression(arr): # Function that makes a linear regression from an 1D array of range 4
    x = range(4)
    A = np.vstack([x,np.ones(len(x))]).T
    m = np.linalg.lstsq(A,arr,rcond=None)
    return m
