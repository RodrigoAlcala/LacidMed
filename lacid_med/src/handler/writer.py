import multiprocessing
import os
import pydicom
import numpy as np
from typing import List
import re


class SingleFileWriter:
    def __init__(self, dicom_file_path: str) -> None:
        try:
            self.dicom_file = pydicom.dcmread(dicom_file_path)
        except (FileNotFoundError, pydicom.errors.InvalidDicomError):
            raise ValueError("Invalid DICOM file or file does not exist.")

    def write(self, new_pixel_array: np.ndarray, output_path: str) -> None:
        if not isinstance(new_pixel_array, np.ndarray):
            raise TypeError("new_pixel_array must be a numpy ndarray")
        if new_pixel_array.ndim != 2:
            raise ValueError("new_pixel_array must have exactly two dimensions")
        ds = self.dicom_file
        ds.PixelData = new_pixel_array.tobytes()
        ds.Rows, ds.Columns = new_pixel_array.shape
        try:
            ds.save_as(output_path)
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error writing file: {str(e)}")


class MultipleFileWriter:
    def __init__(self, dicom_files: List[str]) -> None:
        self.dicom_files = dicom_files

    def write(self, new_pixel_array: np.ndarray, output_path: str) -> None:
        for dicom_file in self.dicom_files:
            try:
                ds = pydicom.dcmread(dicom_file)
                if new_pixel_array.shape == (ds.Rows, ds.Columns):
                    ds.PixelData = new_pixel_array.tobytes()
                    ds.Rows, ds.Columns = new_pixel_array.shape
                    unique_output_path = (
                        output_path + "_" + os.path.basename(dicom_file)
                    )
                    if os.path.exists(unique_output_path) and os.access(
                        unique_output_path, os.W_OK
                    ):
                        ds.save_as(unique_output_path)
                    else:
                        raise ValueError("Invalid or unwritable output path")
                else:
                    raise ValueError(
                        "Shape of new_pixel_array does not match original pixel data"
                    )
            except Exception as e:
                print(f"Error writing file {dicom_file}: {str(e)}")

        def process_file(dicom_file):
            try:
                ds = pydicom.dcmread(dicom_file)
                if new_pixel_array.shape == (ds.Rows, ds.Columns):
                    ds.PixelData = new_pixel_array.tobytes()
                    ds.Rows, ds.Columns = new_pixel_array.shape
                    unique_output_path = (
                        output_path + "_" + os.path.basename(dicom_file)
                    )
                    if os.path.exists(unique_output_path) and os.access(
                        unique_output_path, os.W_OK
                    ):
                        ds.save_as(unique_output_path)
                    else:
                        raise ValueError("Invalid or unwritable output path")
                else:
                    raise ValueError(
                        "Shape of new_pixel_array does not match original pixel data"
                    )
            except Exception as e:
                print(f"Error writing file {dicom_file}: {str(e)}")

        with multiprocessing.Pool() as pool:
            pool.map(process_file, self.dicom_files)


class Deanonymizer:
    def __init__(self, old_directory: str, new_directory: str, output_directory: str) -> None:
        self.old_directory = old_directory
        self.new_directory = new_directory
        self.output_directory = output_directory
        
    def numericalSort(self, value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    def deanonymizer(self, output_name: str = None) -> None:
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)
            print("Output directory created in path.")
        old_files = []
        for file in sorted(os.listdir(self.old_directory), key=self.numericalSort):    
            old_path = self.old_directory + "/" + str(file)
            old_files.append(old_path)
        new_files = []
        for file in sorted(os.listdir(self.new_directory), key=self.numericalSort):    
            new_path = self.new_directory + "/" + str(file)
            new_files.append(new_path)
        if len(old_files) != len(new_files):
            raise ValueError("The number of files in the old and new directories must be the same")
        else:
            for i in range(0,  len(old_files)):    
                if output_name is not None:
                    output_path = self.output_directory + "/" + output_name + "_" + str(i) + ".dcm"
                else:
                    output_path = self.output_directory + "/" + str(i) + ".dcm"
                new_file = pydicom.dcmread(new_files[i])
                new_array = new_file.pixel_array
                writer = SingleFileWriter(old_files[i])
                writer.write(new_array, output_path)