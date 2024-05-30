
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from skimage import data  
from PIL import Image

# lacid_med directory append.
import sys
sys.path.append("C:/Users/santi/Desktop/Fuesmen/code/LacidMed")

from lacid_med.src.handler.converter import Converter

def region_growing(image, seed_point, lower_threshold, upper_threshold):
    # Convert seed point to SimpleITK index
    seed_index = image.TransformPhysicalPointToIndex(seed_point)
    
    # Initialize the segmentation filter
    seg_filter = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    seg_filter.CopyInformation(image)
    seg_filter[seed_index] = 1
    
    # Perform region growing segmentation
    seg_result = sitk.ConfidenceConnected(image, seedList=[seed_index],
                                          numberOfIterations=100,
                                          multiplier=4,
                                          initialNeighborhoodRadius=1,
                                          replaceValue=1)
    
    return seg_result

# Load example image from skimage.data
camera_image = data.camera()

dicom_path = "C:/Users/santi/Desktop/Fuesmen/imagenes/test_volume/test_name_40.dcm"

dicom_image = pydicom.dcmread(dicom_path)
dicom_array = dicom_image.pixel_array
image = sitk.GetImageFromArray(dicom_array)


# Define seed point (you can choose this interactively or programmatically)
seed_point = [50, 50]  # Example seed point coordinates in physical space

# Define lower and upper intensity thresholds
lower_threshold = 100
upper_threshold = 150

# Perform region growing segmentation
segmentation_result = region_growing(image, seed_point, lower_threshold, upper_threshold)

# Convert segmentation result to NumPy array for visualization
segmentation_array = sitk.GetArrayFromImage(segmentation_result)

# Plot the original image and the segmentation result
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(dicom_array, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(segmentation_array, cmap='gray')
axs[1].set_title('Segmentation Result')
axs[1].axis('off')
plt.show()


"""
conversor = Converter(input_directory="C:/Users/santi/Desktop/Fuesmen/imagenes/test_volume", output_directory="C:/Users/santi/Desktop/Fuesmen/imagenes/output_converter")
conversor.dicom_directory_2_jpg(output_name_custom="test_images")
conversor_2 = Converter(input_file= "C:/Users/santi/Desktop/Fuesmen/imagenes/output_converter/test_images_77.JPEG", output_directory="C:/Users/santi/Desktop/Fuesmen/imagenes/output_converter_2")
conversor_2.jpeg_2_nrrd(output_name_custom="test_nrrd")
conversor_3 = Converter(input_directory="C:/Users/santi/Desktop/Fuesmen/imagenes/output_converter", output_directory="C:/Users/santi/Desktop/Fuesmen/imagenes/output_converter_3")
conversor_3.jpeg_directory_2_nrrd()
conversor_4 = Converter(input_directory="C:/Users/santi/Desktop/Fuesmen/imagenes/upscayl_jpg_ultramix_balanced_5x", output_directory="C:/Users/santi/Desktop/Fuesmen/imagenes/output_converter_4")
conversor_4.jpeg_directory_2_nrrd()
"""