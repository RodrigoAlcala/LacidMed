import pydicom
import matplotlib.pyplot as plt
import numpy as np

# lacid_med directory append.
import sys
sys.path.append("C:/Users/santi/Desktop/Fuesmen/code/LacidMed")

from lacid_med.src.processing.operations import Operations
from lacid_med.src.visualization.plotter import DicomPlotter

def main():
    path1 = "C:/Users/santi/Desktop/Fuesmen/imagenes/ZTE_H01_descomp/MR.1.2.840.113619.2.363.10499743.3637646.25026.1684513445.974.90.dcm"
    path2 = "C:/Users/santi/Desktop/Fuesmen/imagenes/correction_N4_original/1.2.840.113619.2.363.10499743.3637646.25026.1684513445.974.90.dcm"

    # carga de los archivos en formato dicom y conversion a array.
    image1 = pydicom.dcmread(path1)
    image2 = pydicom.dcmread(path2)
    image1_array = image1.pixel_array
    image2_array = image2.pixel_array

    path_list = [path1, path2]

    plotter = DicomPlotter(path_list)
    operations = Operations()

    diff = operations.imageDiff(path2, path1, clipping=True)
    plotter._show_image(diff)
    

    

    

    

    




if __name__ == "__main__":
    main()