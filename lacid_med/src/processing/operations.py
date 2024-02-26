import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom


class Operations:
    def __init__(
        self, 
        input_directory: str = None, 
        output_directory: str = None
    ):
        
        """
        Initializes operations object.
        Args:
            input_directory (str): Path to the input directory.
            output_directory (str): Path to the output directory.
        """

        if not isinstance(input_directory, str):
            raise TypeError("input_directory must be a string")
        if not os.path.isdir(input_directory):
            raise ValueError("Invalid directory path: {}".format(input_directory))
        if not os.path.exists(output_directory) or not os.access(output_directory, os.W_OK):
            raise ValueError("Output directory does not exist or is not writable")

        self.input_directory = input_directory
        self.output_directory = output_directory
    

    def imageDiff(self, inputPath1, inputPath2):  
        """
        Substract two images and return the difference in pixel values.

        Args:
            inputPath1 (str): Path to the first image.
            inputPath2 (str): Path to the second image.

        Returns:
            The difference in pixel values between the two images.
        
        """
        image1 = pydicom.dcmread(inputPath1) # path de la imagen 1.
        image2 = pydicom.dcmread(inputPath2) # path de la imagen 2.
        pixelArrayDiff = image2.pixel_array - image1.pixel_array
        pixelArrayDiff = np.abs(pixelArrayDiff)
        return pixelArrayDiff
        
