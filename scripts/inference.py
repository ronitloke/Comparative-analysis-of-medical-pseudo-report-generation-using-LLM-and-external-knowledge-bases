import torch
import numpy as np
from PIL import Image
import pydicom
from io import BytesIO
from torchvision import transforms
from model import load_albef_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer = load_albef_model()
model.load_state_dict(torch.load("albef_model.pth", map_location=device))
model.to(device)
model.eval()

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_dicom(dicom_data):
    dicom_file = pydicom.dcmread(BytesIO(dicom_data))
    image_array = dicom_file.pixel_array
    image = Image.fromarray((image_array / np.max(image_array) * 255).astype(np.uint8)).convert('RGB')
    return image_transforms(image).unsqueeze(0).to(device)

def generate_embeddings(image_tensor, text):
    with torch.no_grad():
        tokenized_text = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids, attention_mask = tokenized_text["input_ids"].to(device), tokenized_text["attention_mask"].to(device)
        image_embed, text_embed = model(image_tensor, input_ids, attention_mask)
    return image_embed.cpu().numpy(), text_embed.cpu().numpy()

def run_inference(dicom_data, text):
    image_tensor = preprocess_dicom(dicom_data)
    return generate_embeddings(image_tensor, text)
