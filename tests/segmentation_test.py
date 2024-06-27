
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from skimage import data  
from PIL import Image

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

dicom_path = "C:/Users/santi/Desktop/Segmented/MR.1.2.840.113619.2.363.10499743.3637646.25026.1684513445.974.70.dcm"
dicom_image = pydicom.dcmread(dicom_path)
dicom_array = dicom_image.pixel_array
filtrator_1 = Filters(sequence_directory=dicom_path)
pre_filt = filtrator_1.apply_minimum_filter(img_arr=dicom_array, kernel_size=6)
segmenter_1 = Segmentation(image_array=dicom_array)
segmentation_array = segmenter_1.region_growing(
            seed_point=[20, 20], 
            multiplier=5,
            number_of_iterations=100, 
            invert_mask=True
            )
filtered_array = np.multiply(segmentation_array, dicom_array)

segmenter_2 = Segmentation(image_array=segmentation_array) 
segmentation_array_2 = segmenter_2.region_growing(
            seed_point=[140, 140], 
            multiplier=3,
            number_of_iterations=1, 
            invert_mask=False
            )


# directory_path = "C:/Users/santi/Desktop/Fuesmen/imagenes/ZTE_H01_descomp"
# loader = DicomLoaderMRI(directory_path=directory_path) 
# vol_array = loader.volumetric_array 
# segmenter_2 = Segmentation(volumetric_array=vol_array)
# segmented_vol = segmenter_2.region_growing(
#             seed_point=[10, 10, 10], 
#             multiplier=5, 
#             number_of_iterations=10, 
#             invert_mask=True
#             )
# filtered_vol = np.multiply(segmented_vol, vol_array)

# sitk.WriteImage(sitk.GetImageFromArray(filtered_vol),"C:/Users/santi/Desktop/segmented_vol.nrrd")
# writer = MultipleFileWriter(loader.sorted_files)
# writer.write(filtered_vol, 'C:/Users/santi/Desktop/Segmented')


# directory_path_2 = "C:/Users/santi/Desktop/Segmented"
# loader_2 = DicomLoaderMRI(directory_path=directory_path_2) 
# vol_array_2 = loader_2.volumetric_array 
# segmenter_2 = Segmentation(volumetric_array=vol_array_2)
# print(vol_array_2.shape)
# segmented_vol_2 = segmenter_2.region_growing(
#             # seed_point=[round(vol_array_2.shape[0]/2), round(vol_array_2.shape[1]/2), round(vol_array_2.shape[2]/2)], 
#             seed_point=[128, 128, 77], 
#             multiplier=5, 
#             number_of_iterations=10, 
#             invert_mask=True
#             )
# filtered_vol_2 = np.multiply(segmented_vol_2, vol_array_2)

# sitk.WriteImage(sitk.GetImageFromArray(filtered_vol_2),"C:/Users/santi/Desktop/segmented_vol_2.nrrd")
# writer_2 = MultipleFileWriter(loader_2.sorted_files)
# writer_2.write(filtered_vol_2, 'C:/Users/santi/Desktop/second_segment')



fig, axs = plt.subplots(1, 4, figsize=(12, 6))
axs[0].imshow(dicom_array, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(pre_filt, cmap='gray')
axs[1].set_title('Pre filter')
axs[1].axis('off')
axs[2].imshow(segmentation_array_2, cmap='gray')
axs[2].set_title('Mask')
axs[2].axis('off')
axs[3].imshow(filtered_array, cmap='gray')
axs[3].set_title('Filtered image')
axs[3].axis('off')
plt.show()

