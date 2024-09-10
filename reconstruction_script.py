import os
import numpy as np
import torch
from model import VQVAE
from PIL import Image
from torchvision import transforms
import argparse
import matplotlib.pyplot as plt

# Define the argument parser
parser = argparse.ArgumentParser()

parser.add_argument("--num_hiddens", type=int, default=128)
parser.add_argument("--num_residual_hiddens", type=int, default=32)
parser.add_argument("--num_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--num_embeddings", type=int, default=512)
parser.add_argument("--commitment_cost", type=float, default=0.25)

args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set the path to the saved model
saved_model_path = "./saved_models/vqvae_10.pth"

# Set the path to the dataset
dataset_path = "D:/downloads/img_align_celeba/img_align_celeba"

# Set the number of images to reconstruct
num_images = 10

# Load the saved model
model = VQVAE(3, args.num_hiddens, args.num_residual_layers, args.num_residual_hiddens, args.num_embeddings, args.embedding_dim, args.commitment_cost).to(device)
model.load_state_dict(torch.load(saved_model_path))
model.eval()

paths = os.listdir("D:/downloads/img_align_celeba/img_align_celeba")
indices = np.random.randint(0, len(paths), 10)
images = []
for i in indices:
    image = Image.open(f"D:/downloads/img_align_celeba/img_align_celeba/{paths[i]}")
    image = image.resize((128, 128))
    image = transforms.ToTensor()(image)
    images.append(image)

# Reconstruct and save the images
images = torch.stack(images).to(device)
with torch.no_grad():
    _, reconstructions, _ = model(images)
    for i, (image, reconstruction) in enumerate(zip(images, reconstructions)):
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(image.cpu().numpy().transpose(1, 2, 0))
        axs[0].set_title("Original Image")
        axs[1].imshow(reconstruction.cpu().numpy().transpose(1, 2, 0))
        axs[1].set_title("Reconstructed Image")

        axs[0].axis("off")
        axs[1].axis("off")

        plt.savefig(f"output/reconstructed_image_{i}.png")
        print(f"Reconstructed image saved to output/reconstructed_image_{i}.png")
