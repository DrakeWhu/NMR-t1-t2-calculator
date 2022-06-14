import os
import numpy as np
from os import listdir
from os.path import isfile, join
import glob
from pathlib import Path
import scipy.io

data_path = Path(r"C:/Users/juanr/Desktop/Universidad/Master/Practicas/programa calculo t1t2/NMR-t1-t2-calculator/rawdata/")
file_list = [str(pp) for pp in data_path.glob("**/*.mat")]

def find_2D_images(): # This function finds if any of the images is a 2D image
   
    images3D = []
    images2D = []

    for i in range(len(file_list)):
        rawData = scipy.io.loadmat(file_list[i])
        if rawData['nPoints'][0][2]:
            images3D.insert(i,rawData['fileName'])
        else:
            images2D.insert(i,rawData['fileName'])
    return images2D

images2D = find_2D_images()

print(images2D)

