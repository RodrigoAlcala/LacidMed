import sys
sys.path.append("/home/clara/PseudoCT/Codigos/LacidMed/lacid_med") #le indica al script de python donde se encuentra el paquete de clases que se utilizarán

import pydicom
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import SimpleITK as sitk

#Se importan clases específicas de módulos dentro del paquete lacid_med
from lacid_med.src.handler.loader import DicomLoaderMRI
from lacid_med.src.processing.filters import Filters
from lacid_med.src.visualization.plotter import DicomPlotter
from lacid_med.src.visualization.histograms import HistogramGenerator 

def main():
    OFFSET=2
    sequence_dir = "/home/clara/PseudoCT/Pacientes/mri_H01/ZTE_H01_descomp"

    loader = DicomLoaderMRI(directory_path=sequence_dir) #se instancia la clase Loader
    org_vol = loader.volumetric_array #raw data organizada
    
    #N4 filter
    filter = Filters(sequence_directory=sequence_dir)
    n4_files = filter.N4_bias_correction_filter(max_iterations=[1,1,1], convergence_threshold=0.001,mask_image=None)
    sitk.WriteImage(sitk.GetImageFromArray(n4_files), "/home/clara/PseudoCT/Codigos/LacidMed/pseudo-CT/output/n4_corrected.nrrd")
    #print(n4_files)

    #Variable de imagenes filtradas 
    image = sitk.ReadImage("/home/clara/PseudoCT/Codigos/LacidMed/pseudo-CT/output/n4_corrected.nrrd") #imagenes filtradas
    vol_filt= sitk.GetArrayFromImage(image) #vol_fit es una matriz 3D de imagenes filtradas


    #Normalizar
    norm_slices_raw = []
    for i in range(np.shape(org_vol)[2]): #se itera en la tercer dimension de la matriz vol_filt
        arr = org_vol[:,:,i] #la variable i recorre todos los cortes
        normalized_arr_raw = filter.normalize_image_filter(image_arr=arr)
        norm_slices_raw.append(normalized_arr_raw) #lista de arrays normalizados

    norm_vol_raw = np.dstack(norm_slices_raw) #dstack apila los arrays

    norm_slices = []
    for i in range(np.shape(vol_filt)[2]): #se itera en la tercer dimension de la matriz vol_filt
        arr = vol_filt[:,:,i] #la variable i recorre todos los cortes
        normalized_arr = filter.normalize_image_filter(image_arr=arr)
        norm_slices.append(normalized_arr) #lista de arrays normalizados


    norm_vol_filt = np.dstack(norm_slices) #dstack apila los arrays

    #Histogram from image raw normalized
    print(np.shape(norm_vol_raw))
    histogram3D = HistogramGenerator(array_3D=norm_vol_raw)
    hist_raw, bins = histogram3D.create_histogram_of_3d_array(offset=OFFSET, show=True, plot_title="Histogram of raw image", xlabel="Pixel value", ylabel="Frequency")

    #Histogram from image filtered
    print(np.shape(norm_vol_filt))
    histogram3D = HistogramGenerator(array_3D=norm_vol_filt)
    hist_filt, bins = histogram3D.create_histogram_of_3d_array(offset=OFFSET, show=True, plot_title="Histogram of filtered image", xlabel="Pixel value", ylabel="Frequency")

    #Histogram peak (normalized)
    index_filt= np.argmax(hist_filt)
    max_value_filt= np.max(hist_filt)
    print('index filt=', index_filt,'value filt=', max_value_filt)

    index_raw= np.argmax(hist_raw)
    max_value_raw= np.max(hist_raw)
    print('index raw=', index_raw,'value raw=', max_value_raw)

    return

if __name__ == "__main__":
    main()