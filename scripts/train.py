import torch
from torch import optim, nn
from torch.nn.utils.rnn import pad_sequence
from dataset import ChestXRayDataset, transform
from model import ALBEFModel, load_albef_model
from gcs_utils import list_and_sort_files

device = "cuda" if torch.cuda.is_available() else "cpu"

def contrastive_loss(image_embed, text_embed, temperature=0.1):
    similarity_matrix = torch.matmul(image_embed, text_embed.t()) / temperature
    labels = torch.arange(image_embed.size(0)).to(device)
    loss_i = nn.CrossEntropyLoss()(similarity_matrix, labels)
    loss_t = nn.CrossEntropyLoss()(similarity_matrix.t(), labels)
    return (loss_i + loss_t) / 2

def train(model, train_dataloader, val_dataloader, epochs=10, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, reports in train_dataloader:
            images = images.to(device)
            tokenized_reports = tokenizer(reports, return_tensors='pt', padding=True, truncation=True, max_length=512)
            input_ids, attention_mask = tokenized_reports["input_ids"].to(device), tokenized_reports["attention_mask"].to(device)

            optimizer.zero_grad()
            image_embed, text_embed = model(images, input_ids, attention_mask)
            loss = contrastive_loss(image_embed, text_embed)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {total_loss/len(train_dataloader):.4f}")

if __name__ == "__main__":
    model, tokenizer = load_albef_model()

    # Load dataset
    train_dataloader, val_dataloader, _ = load_dataloaders()  # Implement this in dataset.py

    train(model, train_dataloader, val_dataloader, epochs=10, lr=1e-4)
    torch.save(model.state_dict(), "albef_model.pth")
    print("Training complete. Model saved.")
