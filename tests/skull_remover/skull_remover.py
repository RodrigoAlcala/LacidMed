import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os

from skimage import data  
from PIL import Image
from scipy.ndimage import label


# lacid_med directory append.
import sys
sys.path.append("C:/Users/santi/Desktop/Fuesmen/code/LacidMed")
from lacid_med.src.handler.loader import DicomLoaderMRI
from lacid_med.src.handler.writer import SingleFileWriter
from lacid_med.src.handler.writer import MultipleFileWriter
from lacid_med.src.handler.writer import Deanonymizer
from lacid_med.src.processing.filters import Filters
from lacid_med.src.visualization.plotter import DicomPlotter
from lacid_med.src.visualization.histograms import HistogramGenerator
from lacid_med.src.processing.fitting import Fitting
from lacid_med.src.processing.operations import Operations
from lacid_med.src.processing.segmentation import Segmentation

directory_path_1 = "C:/Users/santi/Desktop/Fuesmen/imagenes/CASTRO NELIDA FLASH/only_dicom"
#filter_1 = Filters(sequence_directory=directory_path_1)
#n4_files = filter_1.N4_bias_correction_filter(max_iterations=[50,50,50], convergence_threshold=0.001,mask_image=None)
#sitk.WriteImage(sitk.GetImageFromArray(n4_files), "C:/Users/santi/Desktop/Fuesmen/imagenes/n4_corrected.nrrd")
directory_path_2 = "C:/Users/santi/Desktop/Fuesmen/imagenes/n4_corrected.nrrd"
image = sitk.ReadImage("C:/Users/santi/Desktop/Fuesmen/imagenes/n4_corrected.nrrd") #imagenes filtradas
vol_filt= sitk.GetArrayFromImage(image) #vol_fit es una matriz 3D de imagenes filtradas
vol_filt_stack = np.dstack(vol_filt) #dstack apila los arrays

loader_1 = DicomLoaderMRI(directory_path=directory_path_1) 
# vol_array_1 = loader_1.volumetric_array 
vol_array_1 = vol_filt_stack 
segmenter_1 = Segmentation(volumetric_array=vol_array_1)
bg_removed = segmenter_1.background_remover_volumetric(background_seed_point=[10,10,10], background_multiplier=5, background_number_of_iterations=10)
filter_2 = Filters(sequence_directory=directory_path_1)
sobel_filtered = filter_2.sobel_image_filter(img_arr=vol_array_1)
operator_1 = Operations(volumetric_array_1=sobel_filtered)
sobel_scaled = operator_1.scale_matrix_to_value()
segmenter_2 = Segmentation(volumetric_array=sobel_scaled)
sobel_tresholded = segmenter_2.threshold_segmentation(lower_threshold=80)



fig, axs = plt.subplots(1, 3, figsize=(12, 6))
axs[0].imshow(vol_array_1[:, :, 50], cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(sobel_scaled[:, :, 50], cmap='gray')
axs[1].set_title('Image to compare')
axs[1].axis('off')

axs[2].imshow(sobel_tresholded[:, :, 50], cmap='gray')
axs[2].set_title('Image to compare')
axs[2].axis('off')

plt.show()

# codigo para guardar imágenes
sitk.WriteImage(sitk.GetImageFromArray(sobel_tresholded),"C:/Users/santi/Desktop/Fuesmen/imagenes/skull_remover/sobel_tresholded/sobel_tresholded.nrrd")
writer = MultipleFileWriter(loader_1.sorted_files)
writer.write(sobel_tresholded, 'C:/Users/santi/Desktop/Fuesmen/imagenes/skull_remover/sobel_tresholded')


