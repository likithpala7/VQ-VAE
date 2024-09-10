from datasets import load_dataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from dataloader import VAEDataset
from tqdm import tqdm
from model import VQVAE
from torch.optim import Adam
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

parser = argparse.ArgumentParser()

"""
Defining hyperparameters
"""

parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--num_hiddens", type=int, default=128)
parser.add_argument("--num_residual_hiddens", type=int, default=32)
parser.add_argument("--num_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--num_embeddings", type=int, default=512)
parser.add_argument("--commitment_cost", type=float, default=0.25)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Loading the dataset
print("Loading dataset...")
ds = load_dataset("D:/downloads/img_align_celeba/img_align_celeba")
print("Dataset loaded.")

"""
Define transformations and dataloader
"""
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = VAEDataset(ds, transform)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

"""
Define model, optimizer, and loss function
"""

model = VQVAE(3, args.num_hiddens, args.num_residual_layers, args.num_residual_hiddens, args.num_embeddings, args.embedding_dim, args.commitment_cost).to(device)
optimizer = Adam(model.parameters(), lr=args.learning_rate, amsgrad=False)

model.train()
recon_errors = []
perplexities = []
losses = []
for i in range(args.num_epochs):
    batch_recon_errors = []
    batch_perplexities = []
    batch_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {i+1}/{args.num_epochs}"):
        batch = batch.to(device)
        optimizer.zero_grad()
        vq_loss, reconstruction, perplexity = model(batch)
        recon_error = F.mse_loss(reconstruction, batch)
        loss = recon_error + vq_loss
        batch_recon_errors.append(recon_error.item())
        batch_perplexities.append(perplexity.item())
        batch_loss += loss.item()
        loss.backward()
        optimizer.step()

    recon_errors.append(sum(batch_recon_errors)/len(dataloader))
    perplexities.append(sum(batch_perplexities)/len(dataloader))
    batch_loss /= len(dataloader)
    losses.append(batch_loss)
    torch.save(model.state_dict(), f"saved_models/vqvae_{i+1}.pth")
    print(f"Epoch {i+1}, Loss: {batch_loss}")

plt.plot(recon_errors)
plt.title("Reconstruction Errors")
plt.savefig("plots/reconstruction_errors.png")

plt.cla()
plt.plot(perplexities)
plt.title("Perplexities")
plt.savefig("plots/perplexities.png")

plt.cla()
plt.plot(losses)
plt.title("Total Loss")
plt.savefig("plots/total_loss.png")