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
    ) -> sitk.Image:
        """
        Resample the input image to a new spacing.

        Args:
            itk_image: The input image to be resampled.
            new_spacing: The new spacing to be applied.
            is_label: Whether the image is a label image or not.

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

        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkBSpline)

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

    def normalize_image_filter(self, image_arr: np.ndarray, as_array: bool = False):
        """
        Normalize the input image.

        Args:
            image_arr: The input image as a numpy array.
            as_array: Whether to return the normalized image as a numpy array.

        Returns:
            The normalized image as a SimpleITK image or numpy array.
        """
        if not isinstance(image_arr, np.ndarray):
            raise ValueError("Invalid input: image_arr must be a numpy array.")
        image = sitk.GetImageFromArray(image_arr)
        image = sitk.RescaleIntensity(image, 0, 255)
        if as_array:
            image = sitk.GetArrayFromImage(image)
        return image
