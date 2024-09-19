import numpy as np
from torch.utils.data import DataLoader
from dataloader import VAEDataset
from datasets import load_dataset, load_from_disk
from pixelcnn import PixelCNN
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from model import VQVAE
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

"""
Defining hyperparameters
"""

parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--num_hiddens", type=int, default=128)
parser.add_argument("--num_residual_hiddens", type=int, default=32)
parser.add_argument("--num_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--num_embeddings", type=int, default=512)
parser.add_argument("--commitment_cost", type=float, default=0.25)

args = parser.parse_args()

print("Loading dataset...")
# ds = load_dataset("D:/downloads/img_align_celeba/img_align_celeba")
# ds.save_to_disk("D:/downloads/img_align_celeba/img_align_celeba")
ds = load_from_disk("D:/downloads/img_align_celeba/img_align_celeba")
print("Dataset loaded.")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Define dataloader
dataset = VAEDataset(ds, transform)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Load the VQ-VAE model
saved_model = "./saved_models/vqvae_50.pth"
vq_model = VQVAE(3, args.num_hiddens, args.num_residual_layers, args.num_residual_hiddens, args.num_embeddings, args.embedding_dim, args.commitment_cost).cuda()
vq_model.load_state_dict(torch.load(saved_model, weights_only=True))
vq_model.eval()

# Define the PixelCNN model
pixelcnn = PixelCNN().cuda()
optimizer = torch.optim.Adam(pixelcnn.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()
pixelcnn.train()

losses = []
for epoch in range(args.num_epochs):
    batch_loss = 0
    for batch in tqdm(dataloader):
        batch = batch.to("cuda")
        with torch.no_grad():
            # Get the latent codes from the VQ-VAE model
            _, _, _, encoding_indices = vq_model._vq_vae(vq_model._pre_vq_conv(vq_model._encoder(batch)))
        encoding_indices = encoding_indices.view(-1, 32, 32)
        outputs = pixelcnn(encoding_indices).reshape(-1, args.num_embeddings, 32, 32)
        # Loss is calculated by using output of PixelCNN model as the predicted values and the latent codes as the true values
        loss = criterion(outputs, encoding_indices)
        batch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    batch_loss /= len(dataloader)
    losses.append(batch_loss)
    print(f"Epoch {epoch + 1}, Loss: {losses[-1]}")
    torch.save(pixelcnn.state_dict(), f"saved_models/pixelcnn_{epoch+1}.pth")

plt.plot(losses)
plt.savefig("pixelcnn_losses.png")