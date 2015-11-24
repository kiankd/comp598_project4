# Author: Kian Kenyon-Dean
"""
This file contains methods that will take care of the image manipulation due to the
fact that we only have 100 images per class. Thus, we will manipulate each image by
adding the following modified images to our dataset:
    1) For each image, rotate it by d degrees. --> (360/d) * 1500 new images.
    2) For each image we have after rotations, add random noise x times to a 
       random subset of the image's pixels. --> (360/d) * 1500 * x new images.  
"""
