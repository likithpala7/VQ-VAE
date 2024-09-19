import torch
import torch.nn.functional as F
from model import VQVAE
import matplotlib.pyplot as plt

vqvae = VQVAE(3, 128, 2, 32, 512, 64, 0.25).cuda()
vqvae.load_state_dict(torch.load("saved_models/vqvae_10.pth"))
vqvae.eval()

# Assume your codebook has 512 possible embeddings and the latent space is of shape (batch_size, 32, 32)
batch_size = 10
latent_height, latent_width = 32, 32
num_embeddings = 512  # This is the number of latent vectors in the codebook
embedding_dim = 64  # This is the dimensionality of each embedding vector in the codebook

# Step 1: Randomly sample indices from the codebook
random_indices = torch.randint(0, num_embeddings, (batch_size, latent_height, latent_width)).cuda()
ohe_samples = F.one_hot(random_indices.long(), num_embeddings)
quantized = torch.matmul(ohe_samples.float(), vqvae._vq_vae._embedding.weight).permute(0, 3, 1, 2).contiguous()

# Step 4: Pass the embeddings through the decoder
generated_images = vqvae._decoder(quantized)

# Now `generated_images` contains the randomly sampled images from the VQ-VAE
for i, sample in enumerate(generated_images):
    plt.imshow(sample.cpu().detach().numpy().transpose(1, 2, 0))
    plt.axis("off")
    plt.savefig(f"output/random_sample_{i}.png")