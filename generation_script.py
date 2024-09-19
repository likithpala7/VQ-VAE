import torch
from pixelcnn import PixelCNN
from model import VQVAE
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformer import LatentTransformer

pixelcnn = PixelCNN().cuda()
pixelcnn.load_state_dict(torch.load("saved_models/pixelcnn_20.pth"))
pixelcnn.eval()

vqvae = VQVAE(3, 128, 2, 32, 512, 64, 0.25).cuda()
vqvae.load_state_dict(torch.load("saved_models/vqvae_50.pth"))
vqvae.eval()

# Generate samples
num_samples = 10
num_embeddings = 512
height, width = 32, 32
samples = torch.zeros(num_samples, height, width).cuda()

# Get probability distribution of latent codes from PixelCNN model
with torch.no_grad():
    for i in range(height):
        for j in range(width):
            logits = pixelcnn(samples.long())
            probs = torch.softmax(logits[:, :, i, j], dim=-1)
            samples[:, i, j] = torch.multinomial(probs, 1).squeeze(-1)

# Transform the latent codes to images by passing through the VQ-VAE decoder
with torch.no_grad():
    ohe_samples = F.one_hot(samples.long(), num_embeddings)
    quantized = torch.matmul(ohe_samples.float(), vqvae._vq_vae._embedding.weight).permute(0, 3, 1, 2).contiguous()
    samples = vqvae._decoder(quantized)

# Save the samples
for i, sample in enumerate(samples):
    plt.imshow(sample.cpu().numpy().transpose(1, 2, 0))
    plt.axis("off")
    plt.savefig(f"output/sample_{i}.png")