import os
import re
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
import torch
from google.cloud import storage
from torchvision import transforms
from torch.utils.data import Dataset
from io import BytesIO

class ChestXRayDataset(Dataset):
    def __init__(self, dicom_blobs, report_blobs, transform=None):
        self.dicom_blobs = dicom_blobs
        self.report_blobs = report_blobs
        self.transform = transform
        self.client = storage.Client()

        self.dicom_blobs.sort(key=lambda x: x.name)
        self.report_blobs.sort(key=lambda x: x.name)

    def __len__(self):
        return len(self.dicom_blobs)

    def __getitem__(self, idx):
        dicom_blob = self.dicom_blobs[idx]
        dicom_data = dicom_blob.download_as_bytes()
        dicom_file = pydicom.dcmread(BytesIO(dicom_data))
        image_array = dicom_file.pixel_array
        image = Image.fromarray((image_array / np.max(image_array) * 255).astype(np.uint8)).convert('RGB')
        if self.transform:
            image = self.transform(image)

        report_blob = self.report_blobs[idx]
        report_data = report_blob.download_as_text()
        report = re.sub(r'\s+', ' ', report_data.strip())

        return image, report
