"""
The only purpose of this function is quickly identifying the T1 and T2 images using as an advantage that a multi-spin echo test produces images with much bigger width than height
"""

def separate_t1_t2(image_array): # This function separates the images used to calculate T1 and T2
    t1_images = []
    t2_images = []
    for i in range(len(image_array)):
        image = image_array[i]
        width = len(image[0,:])
        if width == 1800: # Here we choose a naïve way of separating: the width must be 5 times the height or bigger. We do this cause mse generates really wide images
            t2_images.insert(i, image)
        else:
            t1_images.insert(i, image)
        i = i+1
    return t1_images, t2_images