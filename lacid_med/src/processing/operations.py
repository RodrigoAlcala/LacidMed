import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom


class Operations:
    def __init__(
        self, 
        input_directory1: str = None,
        input_directory2: str = None, 
        output_directory: str = None
    ):
        
        """
        Initializes operations object.
        Args:
            input_directory1 (str): Path to the input directory 1.
            input_directory2 (str): Path to the input directory 2.
            output_directory (str): Path to the output directory.
        """

        if not isinstance(input_directory1, str) and input_directory1 is not None:
            raise TypeError("input_directory must be a string")
        if input_directory1 is not None:
            if not os.path.isdir(input_directory1):
                raise ValueError("Invalid directory path: {}".format(input_directory1))
        if not isinstance(input_directory2, str) and input_directory2 is not None:
            raise TypeError("input_directory must be a string")
        if input_directory2 is not None:
            if not os.path.isdir(input_directory2):
                raise ValueError("Invalid directory path: {}".format(input_directory2))
        if output_directory is not None:
            if not os.path.exists(output_directory) or not os.access(output_directory, os.W_OK):
                raise ValueError("Output directory does not exist or is not writable")

        self.input_directory1 = input_directory1
        self.input_directory2 = input_directory2
        self.output_directory = output_directory
    

    def imageDiff(self, inputPath1, inputPath2, clipping: bool = False):  
        """
        Substract two images and return the difference in pixel values. If 

        Args:
            inputPath1 (str): Path to the first image.
            inputPath2 (str): Path to the second image.
            clipping (bool): Whether to use clipping to set negative values to 0.

        Returns:
            The difference in pixel values between the two images.
        
        """
        image1 = pydicom.dcmread(inputPath1) # path de la imagen 1.
        image2 = pydicom.dcmread(inputPath2) # path de la imagen 2.
        pixelArrayDiff = image1.pixel_array - image2.pixel_array
        if clipping:
            pixelArrayDiff[pixelArrayDiff < 0] = 0
        else:
            pixelArrayDiff = np.abs(pixelArrayDiff)
        return pixelArrayDiff
        
