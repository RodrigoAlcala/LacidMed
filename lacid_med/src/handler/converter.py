import multiprocessing
import os
import pydicom
import numpy as np
from typing import List
import pathlib
import re 
from PIL import Image

class Converter:
    def __init__(
            self, 
            input_file: str = None,
            input_directory: str = None, 
            output_directory: str = None,
            ) -> None:
        """
        Initializes the Converter instance with the given directory paths.
        Args:
            input_file (str): The path to the DICOM file to be converted.
            input_directory (str): The directory path for the images files to be converted.
            output_directory (str): The directory path for the output converted images files.
        Raises:
            ValueError: If the input_file path is invalid.
            ValueError: If the directory paths are invalid.
            ValueError: If the input_file and input_directory are both None.
        """
        if not input_file and not input_directory:
            raise ValueError("Please provide either input_file or input_directory")
        if input_directory is not None:
            if not os.path.isdir(input_directory):
                raise ValueError("Invalid directory old_directory path")
        if input_file is not None:
            if not os.path.isfile(input_file):
                raise ValueError("Invalid file path")
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
            print("Output directory created in path.")
        self.input_file = input_file
        self.input_directory = input_directory
        self.output_directory = output_directory

    def dicom_2_jpg(self, output_name: str = None) -> None: 
        """
        Converts DICOM files in the input directory to JPEG images and saves them in the output directory.
        Args:
            output_name (str, optional): The name of the output JPEG file. Defaults to None.
        """
        ds = pydicom.dcmread(self.input_directory)
        pixel_array = ds.pixel_array
        pixel_array_normalized = (pixel_array / pixel_array.max()) * 255
        pixel_array_uint8 = pixel_array_normalized.astype('uint8')
        image = Image.fromarray(pixel_array_uint8)
        image.save(self.output_directory) 