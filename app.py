import scipy.signal as sig
import numpy as np
import image_array_creation
import T1_creation
import T2_creation
import separator
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

# Declaring  all the different measurement arrays

t2_banana_d1 = t2[0]
t2_banana_bag_d1 = t2[1]

# Creating the masks for the different fruits

# Creating the T1 images

t1_banana_image = T1_creation.create_T1_image(banana_mask,np.abs(t1_banana_d1))
t1_banana_bag_image = T1_creation.create_T1_image(banana_bag_mask,np.abs(t1_banana_bag_d1))

# Creating the T2 arrays

t2_seedless = t2_oranges[0]
t2_banana_d1 = t2[0]

t2_seedless_3D_array = image_array_creation.create_t2_array(t2_seedless)
t2_banana_3D_array_d1 = image_array_creation.create_t2_array(t2_banana_d1)
 
t2_seedless_mask = np.where(np.abs(t2_seedless_3D_array[0,:,:])<0.0005,0,1)

t2_seedless_image = T2_creation.create_t2_image(t2_seedless_3D_array,t2_seedless_mask)
#t2_banana_image_d1 = create_t2_image(t2_banana_3D_array_d1,banana_mask)

plt.figure(1)
plt.imshow(np.abs(t1_seedless_image), cmap='inferno')
plt.colorbar()
#plt.clim(0.9,1)
plt.show()
