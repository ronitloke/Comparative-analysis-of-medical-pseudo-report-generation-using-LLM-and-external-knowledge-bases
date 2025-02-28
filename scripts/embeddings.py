import torch
import numpy as np
from torchvision import transforms
from model import load_albef_model

device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_albef_model()
model.to(device)
model.eval()

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def generate_image_embeddings(image):
    image = image_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embed, _ = model(image, torch.zeros((1, 512), dtype=torch.int64).to(device), torch.zeros((1, 512), dtype=torch.int64).to(device))
    return image_embed.cpu().numpy().flatten()

def generate_text_embeddings(text):
    tokenized_text = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = tokenized_text["input_ids"].to(device)
    attention_mask = tokenized_text["attention_mask"].to(device)
    with torch.no_grad():
        _, text_embed = model(torch.zeros((1, 3, 224, 224)).to(device), input_ids, attention_mask)
    return text_embed.cpu().numpy().flatten()
