import uuid
import os
import pydicom
from typing import List


class Anonymizer:
    def __init__(self, dicom_files: List[str], output_dir: str) -> None:
        """
        Initializes the Anonymizer object with the provided DICOM file paths and output directory.

        Args:
            dicom_files (List[str]): A list of DICOM file paths to be anonymized.
            output_dir (str): The directory where the anonymized DICOM files will be saved.

        Raises:
            ValueError: If the output directory does not exist or is not writable.
        """
        self.dicom_files = dicom_files
        self.output_dir = output_dir
        if not os.path.exists(output_dir) or not os.access(output_dir, os.W_OK):
            raise ValueError("Output directory does not exist or is not writable")

    def anonymize(self) -> None:
        """
        Calls the _perform_anonymization method to perform the anonymization process for each DICOM file in the input list.

        Raises:
            FileNotFoundError: If a DICOM file is not found.
            PermissionError: If there is a permission issue accessing the DICOM files.
        """
        try:
            self._perform_anonymization()
        except (FileNotFoundError, PermissionError) as e:
            raise e

    def _perform_anonymization(self) -> None:
        """
        Reads each DICOM file, replaces the patient ID with a randomly generated UUID, removes private tags and sequences,
        and saves the anonymized DICOM files in the output directory.
        """
        study_id = str(uuid.uuid4())
        for dicom_file in self.dicom_files:
            if os.path.exists(dicom_file):
                ds = pydicom.dcmread(dicom_file)
                ds.PatientID = study_id
                ds.remove_private_tags()
                ds.remove_private_sequence()
                output_file = os.path.join(
                    self.output_dir, f"anonymized_{os.path.basename(dicom_file)}"
                )
                ds.save_as(output_file)
            else:
                raise FileNotFoundError(f"File not found: {dicom_file}")
