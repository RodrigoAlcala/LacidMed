import sys
sys.path.append("/home/clara/PseudoCT/Codigos/LacidMed/lacid_med") #le indica al script de python donde se encuentra el paquete de clases que se utilizarán

import pydicom
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.optimize import curve_fit

#Se importan clases específicas de módulos dentro del paquete lacid_med
from lacid_med.src.handler.loader import DicomLoaderMRI
from lacid_med.src.processing.filters import Filters
from lacid_med.src.visualization.plotter import DicomPlotter
from lacid_med.src.visualization.histograms import HistogramGenerator 


def gaussian(x, mean, amplitude, standard_deviation):
    """
    Gaussian function.
    
    Parameters:
        x (float or numpy.ndarray): Input value(s).
        mean (float): Mean of the Gaussian distribution.
        amplitude (float): Amplitude of the Gaussian curve.
        standard_deviation (float): Standard deviation of the Gaussian distribution.
        
    Returns:
        float or numpy.ndarray: Value(s) of the Gaussian function evaluated at input x.
    """
    return amplitude * np.exp(-((x - mean) / standard_deviation) ** 2 / 2)

def fit_gaussian_to_histogram(hist, bins, initial_guess= [0, 0, 0]):
    """
    Fit a Gaussian curve to the given histogram.
    
    Parameters:
        hist (numpy.ndarray): Image histogram.
        bins (numpy.ndarray): Bin edges of the histogram.
        
    Returns:
        tuple: Parameters of the Gaussian fit (mean, amplitude, standard deviation).
    """
    # Calculate bin centers
    #bin_centers = (bins[:-1] + bins[1:]) / 2
    #print('bin centers=bin_centers', len(bin_centers), 'hist=',len(hist))

    # Perform curve fitting
    popt, _ = curve_fit(gaussian, bins , hist, p0=initial_guess)

    # Return the fitted parameters
    return popt


def n_polynomial_fit(x,y,n):
    coefs=np.polyfit(x,y,n)
    pol_func=np.polyval(coefs,x)
    return pol_func


def main():
    OFFSET=3
    sequence_dir = "/home/clara/PseudoCT/Pacientes/mri_H01/ZTE_H01_descomp"

    loader = DicomLoaderMRI(directory_path=sequence_dir) #se instancia la clase Loader
    org_vol = loader.volumetric_array #raw data organizada
    
    #N4 filter
    filter = Filters(sequence_directory=sequence_dir)
    #n4_files = filter.N4_bias_correction_filter(max_iterations=[1,1,1], convergence_threshold=0.001,mask_image=None)
    #sitk.WriteImage(sitk.GetImageFromArray(n4_files), "/home/clara/PseudoCT/Codigos/LacidMed/pseudo-CT/output/n4_corrected.nrrd")
    #print(n4_files)

    #Variable de imagenes filtradas 
    image = sitk.ReadImage("/home/clara/PseudoCT/Codigos/LacidMed/pseudo-CT/output/n4_corrected.nrrd") #imagenes filtradas
    vol_filt= sitk.GetArrayFromImage(image) #vol_fit es una matriz 3D de imagenes filtradas
    
    vol_filt_stack = np.dstack(vol_filt) #dstack apila los arrays
    sitk.WriteImage(sitk.GetImageFromArray(vol_filt_stack),"/home/clara/PseudoCT/Codigos/LacidMed/pseudo-CT/output/no_norm_filt.nrrd")

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

    sitk.WriteImage(sitk.GetImageFromArray(norm_vol_filt),"/home/clara/PseudoCT/Codigos/LacidMed/pseudo-CT/output/filtrada_norm.nrrd")

    #Histogram from image raw normalized
    #print(np.shape(norm_vol_raw))
    histogram3D = HistogramGenerator(array_3D=norm_vol_raw)
    hist_raw, bins = histogram3D.create_histogram_of_3d_array(offset=OFFSET, show=True, plot_title="Histogram of raw image", xlabel="Pixel value", ylabel="Frequency")

    #Histogram from image filtered
    #print(np.shape(norm_vol_filt))
    histogram3D = HistogramGenerator(array_3D=norm_vol_filt)
    hist_filt, bins = histogram3D.create_histogram_of_3d_array(offset=OFFSET, show=True, plot_title="Histogram of filtered image", xlabel="Pixel value", ylabel="Frequency")
    
    #print('hit_filt=', hist_filt, 'bins=', bins)

    #Histogram peak (normalized)
    index_filt= np.argmax(hist_filt)
    max_value_filt= np.max(hist_filt)
    #print('index filt=', index_filt,'value filt=', max_value_filt)

    index_raw= np.argmax(hist_raw)
    max_value_raw= np.max(hist_raw)
    #print('index raw=', index_raw,'value raw=', max_value_raw)

    #Fiteo first gaussian
    gaussian_fit_1= fit_gaussian_to_histogram(hist=hist_filt[:len(hist_filt/2)], bins=bins[:len(bins/2)], initial_guess=[12, 3600, 0.5])
   
    #Fiteo second gaussian
    n=9
    poly_fit_2= n_polynomial_fit(bins[10:36], hist_filt[10:36],n)

    #Fiteo third gaussian
    gaussian_fit_3= fit_gaussian_to_histogram(hist=hist_filt[36:65], bins=bins[36:65], initial_guess=[50, 25, 0.25])
   
    #Fiteo fourth gaussian
    gaussian_fit_4= fit_gaussian_to_histogram(hist=hist_filt[90:], bins=bins[90:], initial_guess=[150, 275, 0.3])

    #Ploteo de histogramas y fiteos
    print(len(gaussian_fit_1), len(gaussian_fit_3), len(gaussian_fit_4), len(poly_fit_2))
    plt.figure(figsize=(8, 6))
    plt.plot(bins, hist_filt, label='Histogram', color='blue')
    plt.plot(bins[0:10], gaussian(bins[0:10], *gaussian_fit_1),"x", label='First Gaussian Fit', color='red')
    plt.plot(bins[10:36], poly_fit_2,"x", label='Polynomial Fit', color='purple')
    plt.plot(bins[36:65], gaussian(bins[36:65], *gaussian_fit_3),"x", label='Second Gaussian Fit', color='green',)
    plt.plot(bins[90:], gaussian(bins[90:], *gaussian_fit_4),"x", label='Third Gaussian Fit', color='orange')
    
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title ('ZTE Fit to Histogram')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()