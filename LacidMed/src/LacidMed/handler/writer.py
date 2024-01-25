import multiprocessing
import os
import pydicom
import numpy as np
from typing import List


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
                    unique_output_path = output_path + "_" + os.path.basename(dicom_file)
                    if os.path.exists(unique_output_path) and os.access(unique_output_path, os.W_OK):
                        ds.save_as(unique_output_path)
                    else:
                        raise ValueError("Invalid or unwritable output path")
                else:
                    raise ValueError("Shape of new_pixel_array does not match original pixel data")
            except Exception as e:
                print(f"Error writing file {dicom_file}: {str(e)}")

        def process_file(dicom_file):
            try:
                ds = pydicom.dcmread(dicom_file)
                if new_pixel_array.shape == (ds.Rows, ds.Columns):
                    ds.PixelData = new_pixel_array.tobytes()
                    ds.Rows, ds.Columns = new_pixel_array.shape
                    unique_output_path = output_path + "_" + os.path.basename(dicom_file)
                    if os.path.exists(unique_output_path) and os.access(unique_output_path, os.W_OK):
                        ds.save_as(unique_output_path)
                    else:
                        raise ValueError("Invalid or unwritable output path")
                else:
                    raise ValueError("Shape of new_pixel_array does not match original pixel data")
            except Exception as e:
                print(f"Error writing file {dicom_file}: {str(e)}")

        with multiprocessing.Pool() as pool:
            pool.map(process_file, self.dicom_files)