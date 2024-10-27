# Common libraries for medical image processing
import pydicom  # for DICOM files
import nibabel as nib  # for neuroimaging data
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_dicom_series(directory):
    """
    Load a series of DICOM images and sort them by slice location
    """
    # Get all DICOM files in directory
    dicom_files = list(Path(directory).glob('*.dcm'))
    
    # Read all files
    slices = [pydicom.dcm.read_file(str(f)) for f in dicom_files]
    
    # Sort slices by Instance Number or Slice Location
    slices.sort(key=lambda x: float(x.InstanceNumber))
    
    # Extract pixel arrays
    image_3d = np.stack([s.pixel_array for s in slices])
    
    return image_3d, slices

def display_datscan_slice(image_3d, slice_idx, title="DaTSCAN slice"):
    """
    Display a single slice with proper visualization settings
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image_3d[slice_idx], cmap='hot', aspect='equal')
    plt.colorbar(label='Intensity')
    plt.title(f"{title} - Slice {slice_idx}")
    plt.axis('off')
    plt.show()

def get_dicom_metadata(dicom_slice):
    """
    Extract relevant metadata from DICOM file
    """
    metadata = {
        'PatientID': dicom_slice.PatientID,
        'StudyDate': dicom_slice.StudyDate,
        'Modality': dicom_slice.Modality,
        'SliceThickness': dicom_slice.SliceThickness,
        'PixelSpacing': dicom_slice.PixelSpacing,
        # Add other relevant metadata fields
    }
    return metadata

# Example usage
directory = 'Images'
image_3d, dicom_slices = load_dicom_series(directory)

# Display middle slice
middle_slice = image_3d.shape[0] // 2
display_datscan_slice(image_3d, middle_slice)

# Print metadata from middle slice
metadata = get_dicom_metadata(dicom_slices[middle_slice])
print("\nDICOM Metadata:")
for key, value in metadata.items():
    print(f"{key}: {value}")