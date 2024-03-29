import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom


class Operations:
    def __init__(
        self, 
        volumetric_array_1: np.ndarray = None,
        volumetric_array_2: np.ndarray = None, 
    ):
        """
        Initializes operations object.
        Args:
            volumetric_array_1 (np.ndarray): first volumetric array.
            volumetric_array_2 (np.ndarray): second volumetric array.
        Raises:
            TypeError: If volumetric_array_1 or volumetric_array_2 is not a numpy array.
        """
        if not isinstance(volumetric_array_1, np.ndarray) and volumetric_array_1 is not None:
            raise TypeError("volumetric_array_1 must be a numpy array")
        
        if not isinstance(volumetric_array_2, np.ndarray) and volumetric_array_2 is not None:
            raise TypeError("volumetric_array_2 must be a numpy array")
        
        self.volumetric_array_1 = volumetric_array_1
        self.volumetric_array_2 = volumetric_array_2
    

    def image_difference(self, pixel_array_1: np.ndarray, pixel_array_2: np.ndarray, clipping: bool = False):  
        """
        Substract two images and return the difference in pixel values. If clipping is set to True, negative values are set to 0.
        Args:
            pixel_array_1 (np.ndarray): pixel array of the first image.
            pixel_array_2 (np.ndarray): pixel array of the second image.
            clipping (bool, optional): Whether to use clipping to set negative values to 0.

        Returns:
            The difference in pixel values between the two images.
        """
        if pixel_array_1.shape != pixel_array_2.shape:
            raise ValueError("pixel arrays must have the same shape")
        else:
            pixel_array_diff = pixel_array_1 - pixel_array_2
            if clipping:
                pixel_array_diff[pixel_array_diff < 0] = 0
            else:
                pixel_array_diff = np.abs(pixel_array_diff)
        return pixel_array_diff
    
    def volume_difference(self, image_number: int = None, clipping: bool = False):
        """ 
        Substract the second volumetric array from the first volumetric array and return the difference in pixel values, slice per slice. 
        If clipping is set to True, negative values are set to 0.
        Args:
            image_number (int, optional): The number of the image to use in case a single image is needed.
            clipping (bool, optional): Whether to use clipping to set negative values to 0. Defaults to False.

        Returns:
            The difference in pixel values between the two images.
        """
        if self.volumetric_array_1.shape != self.volumetric_array_2.shape:
            raise ValueError("volumetric arrays must have the same shape")
        else:
            if image_number is None:
                vol_array_diff = self.volumetric_array_1 - self.volumetric_array_2
                if clipping:
                    vol_array_diff[vol_array_diff < 0] = 0
                else:
                    vol_array_diff = np.abs(vol_array_diff)
                return vol_array_diff
            elif image_number not in range(0, self.volumetric_array_1.shape[2]):
                raise ValueError("image_number must be in range 0 to " + str(self.volumetric_array_1.shape[2]))
            elif image_number is not None:
                pixel_array_1 = self.volumetric_array_1[:, :,image_number]
                pixel_array_2 = self.volumetric_array_2[:, :,image_number]
                pixel_array_diff = self.image_difference(pixel_array_1, pixel_array_2, clipping)
            return pixel_array_diff
        

    def threshold_segmentation(self, image_array: np.array = None, lower_threshold: int = None, upper_threshold: int = None):
        """
        Generate a segmentation of the array based on a threshold. 
        If no volumetric array is given, volumetric_array_1 is used.
        Args:
            image_array (np.array, optional): The image or volume array to segment. Defaults to volumetric_array_1.
            lower_threshold (int, optional): The lower threshold for the segmentation. Defaults to the minimum value in the array.
            upper_threshold (int, optional): The upper threshold for the segmentation. Defaults to the maximum value in the array.
        """
        if lower_threshold == None: 
            lower_threshold = 0 
            print("Check lower threshold")
        if upper_threshold >= np.max(self.array_3D):
            raise ValueError("Upper threshold must be less than the maximum value in the array.")
        if upper_threshold == None: 
            upper_threshold = np.max(self.array_3D)
            print("Check upper threshold")
        if lower_threshold < np.min(self.array_3D):
            raise ValueError("Lower threshold must be greater than the minimum value in the array.")
        if upper_threshold < lower_threshold:
            raise ValueError("Upper threshold must be greater than the lower threshold.")
        if image_array == None:
            image_array = self.volumetric_array_1
        clipped_array = np.clip(image_array, lower_threshold, upper_threshold)
        return clipped_array
    