import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantize(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantize, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encoding_indices
    
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(False),
            nn.Conv2d(in_channels, num_residual_hiddens, 3, 1, 1),
            nn.ReLU(False),
            nn.Conv2d(num_residual_hiddens, num_hiddens, 1)
        )

    def forward(self, x):
        return x + self._block(x)
    
class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens) for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
    
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels, num_hiddens//2, 4, 2, 1)
        self._conv_2 = nn.Conv2d(num_hiddens//2, num_hiddens, 4, 2, 1)
        self._conv_3 = nn.Conv2d(num_hiddens, num_hiddens, 3, 1, 1)
        self._residual_stack = ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        x = torch.relu(self._conv_1(x))
        x = torch.relu(self._conv_2(x))
        x = torch.relu(self._conv_3(x))
        return self._residual_stack(x)
    
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels, num_hiddens, 3, 1, 1)
        self._residual_stack = ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._deconv_1 = nn.ConvTranspose2d(num_hiddens, num_hiddens//2, 4, 2, 1)
        self._deconv_2 = nn.ConvTranspose2d(num_hiddens//2, 3, 4, 2, 1)

    def forward(self, x):
        x = self._conv_1(x)
        x = self._residual_stack(x)
        x = torch.relu(self._deconv_1(x))
        x = torch.sigmoid(self._deconv_2(x))
        return x
    
class VQVAE(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        self._encoder = Encoder(in_channels, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(num_hiddens, embedding_dim, 1)
        self._vq_vae = VectorQuantize(num_embeddings, embedding_dim, commitment_cost)
        self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, encodings = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity