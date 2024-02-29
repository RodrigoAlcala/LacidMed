import numpy as np
import os
import pydicom
from typing import List
import SimpleITK as sitk


class Filters:
    def __init__(
        self,
        sequence_directory: str,
        dicom_files: List[pydicom.Dataset],
        output_dir: str,
    ):
        """
        Initialize Filters object.
        Args:
            sequence_directory (str): The directory path for the sequence.
            dicom_files (List[pydicom.Dataset]): List of DICOM files.
            output_dir (str): The output directory path.
        Raises:
            ValueError: If `sequence_directory` is not a valid directory path.
            ValueError: If `dicom_files` is an empty list.
            ValueError: If any item in `dicom_files` is not an instance of `pydicom.Dataset`.
            TypeError: If `sequence_directory` is not a string.
            TypeError: If `dicom_files` is not a list.
            ValueError: If `output_dir` does not exist or is not writable.
            TypeError: If `output_dir` is not a string.
        """
        if not isinstance(sequence_directory, str):
            raise TypeError("sequence_directory must be a string")
        if not os.path.isdir(sequence_directory):
            raise ValueError("Invalid directory path: {}".format(sequence_directory))
        if not isinstance(dicom_files, list):
            raise TypeError("`dicom_files` must be a list.")
        for file in dicom_files:
            if not isinstance(file, pydicom.Dataset):
                raise ValueError(
                    "All elements in `dicom_files` must be instances of `pydicom.Dataset`."
                )
        if not isinstance(output_dir, str):
            raise TypeError("output_dir must be a string")
        if not os.path.exists(output_dir) or not os.access(output_dir, os.W_OK):
            raise ValueError("Output directory does not exist or is not writable")

        self.output_dir = output_dir
        self.dicom_files = dicom_files
        self.sequence_directory = sequence_directory

    def resample_image(
        self,
        itk_image,
        new_spacing: List[float] = [1.0, 1.0, 1.0],
        is_label: bool = False,
        interpolator: int = sitk.sitkBSpline
    ) -> sitk.Image:
        """
        Resample the input image to a new spacing.

        Args:
            itk_image: The input image to be resampled.
            new_spacing: The new spacing to be applied.
            is_label: Whether the image is a label image or not.
            interpolator: The interpolator to be used for resampling.

        Returns:
            The resampled image.
        """
        original_spacing = itk_image.GetSpacing()
        original_size = itk_image.GetSize()

        new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
        ]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(itk_image.GetDirection())
        resampler.SetOutputOrigin(itk_image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(itk_image.GetPixelIDValue())
        resampler.SetInterpolator(interpolator)

        return resampler.Execute(itk_image)

    def N4_bias_correction_filter(
        self,
        mask_image=None,
        convergence_threshold: float = 0.001,
        max_iterations: List[int] = [3, 3, 3],
    ) -> sitk.Image:
        """
        Apply N4 bias correction to the input image.

        Args:
            mask_image: The mask image to be used for bias correction.
            convergence_threshold: The convergence threshold for the bias correction algorithm.
            max_iterations: The maximum number of iterations for the bias correction algorithm.

        Returns:
            The bias-corrected image.
        """
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(self.sequence_directory))
        image = reader.Execute()
        image = sitk.Cast(image, sitk.sitkFloat32)

        image = self.resample_image(image)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(max_iterations)
        corrector.SetConvergenceThreshold(convergence_threshold)

        if mask_image is not None:
            mask = sitk.ReadImage(mask_image)
            n4_filtered_image = corrector.Execute(image, mask)
        else:
            mask = sitk.OtsuThreshold(image, 0, 1, 200)
            n4_filtered_image = corrector.Execute(image, mask)

        log_bias_field = corrector.GetLogBiasFieldAsImage(image)
        corrected_image_full_res = n4_filtered_image / sitk.Exp(log_bias_field)

        sitk.WriteImage(
            corrected_image_full_res,
            os.path.join(self.output_dir, "n4_corrected.nrrd"),
        )

        return corrected_image_full_res

    def normalize_image_filter(self, image_arr: np.ndarray):
        """
        Normalize the input image.

        Args:
            image_arr: The input image as a numpy array.

        Returns:
            The normalized image as a SimpleITK image.
        """
        if not isinstance(image_arr, np.ndarray):
            raise ValueError("Invalid input: image_arr must be a numpy array.")
        image = sitk.GetImageFromArray(image_arr)
        image = sitk.RescaleIntensity(image, 0, 255)
        return image
    
    def gaussian_image_filter(self, img_arr: np.ndarray) -> sitk.Image:
        """
        Apply a Gaussian filter to the input image.

        Args:
            img_arr: The input image as a numpy array.

        Returns:
            The filtered image as a SimpleITK image.
        """
        if not isinstance(img_arr, np.ndarray):
            raise ValueError("Invalid input: img_arr must be a numpy array.")
        image = sitk.GetImageFromArray(img_arr)
        image = sitk.SmoothingRecursiveGaussian(image, sigma=1.0)
        return image

    def median_image_filter(self, img_arr: np.ndarray):
        """
        Apply median filter to the input image.

        Args:
            img_arr: The input image as a numpy array. It should have shape (height, width) and data type uint8.

        Returns:
            The filtered image as a SimpleITK image.
        """
        if not isinstance(img_arr, np.ndarray):
            raise ValueError("Invalid input: img_arr must be a numpy array.")
        if img_arr.size == 0:
            raise ValueError("Invalid input: img_arr is empty.")
        try:
            image = sitk.GetImageFromArray(img_arr)
            image = sitk.MedianImageFilter().Execute(image)
            return image
        except Exception as e:
            raise ValueError("Error during median image filtering: {}".format(str(e)))
    
    def binary_threshold(self, img_arr: np.ndarray, lower_threshold: int = 0, upper_threshold: int = 1, inside_value: int = 1, outside_value: int = 0):
        """
        Apply binary thresholding to the input image array.

        Args:
            img_arr (np.ndarray): The input image array.
            lower_threshold (int): The lower threshold value (default is 0).
            upper_threshold (int): The upper threshold value (default is 1).
            inside_value (int): The value to assign to pixels within the threshold range (default is 1).
            outside_value (int): The value to assign to pixels outside the threshold range (default is 0).

        Returns:
            The thresholded image.
        """
        if not isinstance(img_arr, np.ndarray):
            raise ValueError("Invalid input: img_arr must be a numpy array.")
        if lower_threshold > upper_threshold:
            raise ValueError("Invalid threshold values: lower_threshold must be less than or equal to upper_threshold.")
        try:
            image = sitk.GetImageFromArray(img_arr)
            thresholder = sitk.BinaryThresholdImageFilter()
            thresholder.SetLowerThreshold(lower_threshold)
            thresholder.SetUpperThreshold(upper_threshold)
            thresholder.SetInsideValue(inside_value)
            thresholder.SetOutsideValue(outside_value)
            image = thresholder.Execute(image)
            return image
        except Exception as e:
            raise ValueError("Error during binary thresholding: {}".format(str(e)))

    def sobel_image_filter(self, img_arr: np.ndarray):
        """
        Apply Sobel edge detection filter to the input image array.

        Args:
            img_arr (np.ndarray): The input image array.

        Returns:
            np.ndarray: The filtered image array.
        """
        if not isinstance(img_arr, np.ndarray):
            raise ValueError("Invalid input: img_arr must be a numpy array.")

        if img_arr.size == 0:
            raise ValueError("Invalid input: img_arr is empty.")

        if len(img_arr.shape) != 2 and len(img_arr.shape) != 3:
            raise ValueError("Invalid input: img_arr must be a 2D or 3D numpy array.")

        if img_arr.dtype != np.uint8:
            raise ValueError("Invalid input: img_arr must have data type uint8.")

        try:
            image = sitk.GetImageFromArray(img_arr)
            image = sitk.SobelEdgeDetectionImageFilter().Execute(image)
            return image
        except Exception as e:
            raise ValueError("Error during Sobel image filtering: {}".format(str(e)))
    
    def laplacian_image_filter(self, img_arr: np.ndarray):
        """
        Apply Laplacian image filter to the input image array.

        Args:
            img_arr (np.ndarray): Input image array.

        Returns:
            np.ndarray: Filtered image array.
        """
        if not isinstance(img_arr, np.ndarray):
            raise ValueError("Invalid input: img_arr must be a numpy array.")

        if img_arr.size == 0:
            raise ValueError("Invalid input: img_arr is empty.")

        if len(img_arr.shape) != 2 and len(img_arr.shape) != 3:
            raise ValueError("Invalid input: img_arr must be a 2D or 3D numpy array.")

        if img_arr.dtype != np.uint8:
            raise ValueError("Invalid input: img_arr must have data type uint8.")

        try:
            image = sitk.GetImageFromArray(img_arr)
            image = sitk.LaplacianImageFilter().Execute(image)
            return image
        except Exception as e:
            raise ValueError("Error during Laplacian image filtering: {}".format(str(e)))