import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom


class Operations:
    def __init__(self, input_directory: str = None, output_directory: str = None):
        self.input_directory = input_directory
        self.output_directory = output_directory
    

    def imageDiff(self, inputPath1, inputPath2):  
        image1 = pydicom.dcmread(inputPath1) # path de la imagen original.
        image2 = pydicom.dcmread(inputPath2) # path de la imagen procesada.
        pixelArrayDiff = image2.pixel_array - image1.pixel_array
        pixelArrayDiff[pixelArrayDiff<0] = 0
        plt.subplot(131) 
        plt.title('Imagen original')
        plt.imshow(image1.pixel_array)
        plt.subplot(132) 
        plt.title('Imagen procesada')
        plt.imshow(image2.pixel_array)
        plt.subplot(133)
        plt.title('Diferencia entre imágenes')
        plt.imshow(pixelArrayDiff) 
        plt.show()
