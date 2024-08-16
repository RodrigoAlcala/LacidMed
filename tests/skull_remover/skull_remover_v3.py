import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from skimage import data  
from PIL import Image
from scipy.ndimage import label
from skimage import data, img_as_float
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.draw import circle_perimeter

# lacid_med directory append.
import sys
sys.path.append("C:/Users/santi/Desktop/Fuesmen/code/LacidMed")
from lacid_med.src.handler.loader import DicomLoaderMRI
from lacid_med.src.handler.writer import SingleFileWriter, MultipleFileWriter, Deanonymizer
from lacid_med.src.processing.filters import Filters
from lacid_med.src.visualization.plotter import DicomPlotter
from lacid_med.src.visualization.histograms import HistogramGenerator
from lacid_med.src.processing.fitting import Fitting
from lacid_med.src.processing.operations import Operations
from lacid_med.src.processing.segmentation import Segmentation


# Set directory path
directory_path_1 = "C:/Users/santi/Desktop/Fuesmen/imagenes/CASTRO NELIDA FLASH/S0001952/SE020097"

# Load MRI data
loader_1 = DicomLoaderMRI(directory_path=directory_path_1) 
vol_array_1 = loader_1.volumetric_array 
scaler_1 = Operations(volumetric_array_1=vol_array_1)
vol_array_1 = scaler_1.scale_matrix_to_value()

# Background removal
segmenter_1 = Segmentation(volumetric_array=vol_array_1)
bg_removed = segmenter_1.background_remover_volumetric(background_seed_point=[10,10,10], background_multiplier=3, background_number_of_iterations=10)
# Normalize the pixel data to the range [0, 255]
image = bg_removed
image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
image = np.uint8(image)
# Apply the Sobel filter
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
sobel_magnitude = np.uint8(np.absolute(sobel_magnitude))
# Optional: Normalize the Sobel output
sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)



# Visualization
fig, axs = plt.subplots(1, 3, figsize=(12, 6))
axs[0].imshow(vol_array_1[:, :, 80], cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(bg_removed[:, :, 80], cmap='gray')
axs[1].set_title('Image 1')
axs[1].axis('off')

axs[2].imshow(sobel_magnitude[:, :, 80], cmap='gray')
axs[2].set_title('Image 2')
axs[2].axis('off')



plt.show()

# Save images (optional)
# sitk.WriteImage(sitk.GetImageFromArray(sobel_tresholded), "C:/Users/santi/Desktop/Fuesmen/imagenes/skull_remover/sobel_tresholded/sobel_tresholded.nrrd")
# writer = MultipleFileWriter(loader_1.sorted_files)
# writer.write(tresholded, 'C:/Users/santi/Desktop/Fuesmen/imagenes/skull_remover/tresholded')
