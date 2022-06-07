"""
This file creates the evolution of T1 for a given array of T1 images
"""

import numpy as np

def average(image): # This function takes a 60x60 image and returns the average T1 of the whole image

    sum = 0
    total_pixels = 0

    for i in range(60):
        for j in range(60):
            if image[i, j] != 0:
                total_pixels += 1
                sum += abs(image[i,j])
    
    t1_average = sum/total_pixels
    return t1_average



def evolution(array): # This function takes an array of 60x60 images and makes an array of the averages of each one, ordered as they where before

    average_t1_array = []

    for i in range(3):
        mean = average(abs(array[i,:,:]))
        average_t1_array.append(mean)
    
    return average_t1_array

def linear_regression(arr): # Function that makes a linear regression from an 1D array of range 4
    x = range(4)
    A = np.vstack([x,np.ones(len(x))]).T
    m = np.linalg.lstsq(A,arr,rcond=None)
    return m
