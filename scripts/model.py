import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torchvision.models import resnet50

class ALBEFModel(nn.Module):
    def __init__(self, image_model, text_model, embed_dim=512):
        super(ALBEFModel, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.image_proj = nn.Linear(image_model.fc.in_features, embed_dim)
        self.text_proj = nn.Linear(text_model.config.hidden_size, embed_dim)

    def forward(self, image, text_input_ids, text_attention_mask):
        image_features = self.image_model(image)
        image_embed = self.image_proj(image_features)

        text_output = self.text_model(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_features = text_output.last_hidden_state[:, 0, :]
        text_embed = self.text_proj(text_features)

        return image_embed, text_embed
