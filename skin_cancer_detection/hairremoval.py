import cv2
import matplotlib.pyplot as plt
import numpy as np

# function to remove hair based on filtering
def hair_removal(image):
    # converts image to grayScale
    grayScale = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2GRAY)
    # kernel for morphologyEx
    kernel = cv2.getStructuringElement(1,(10,10))
    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(np.float32(grayScale), cv2.MORPH_BLACKHAT, kernel)
    # apply thresholding to blackhat to focus on hairs
    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    # inpaint with original image and threshold image to fill in area behind hair
    final_image = cv2.inpaint(np.float32(image),threshold,5,cv2.INPAINT_TELEA)

    return final_image
