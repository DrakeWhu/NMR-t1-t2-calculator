# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 18:30:59 2022

@author: josal
"""

import scipy.signal as sig
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


rawData = scipy.io.loadmat('./data/Old_RARE.2022.03.29.17.27.58.mat')
imagen = rawData['image3D']
imagen2d = imagen[5,:,:]

plt.figure(1)
plt.imshow(np.abs(imagen2d), cmap='gray')
plt.show()