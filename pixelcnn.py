import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedCNN(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedCNN, self).__init__(*args, **kwargs)
        assert mask_type in ['A', 'B'], "Mask type must be 'A' or 'B'"
        self.mask_type = mask_type
        self.register_buffer('mask', self.weight.data.clone())
        _, _, h, w = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, h // 2, w // 2 + (self.mask_type == 'B'):] = 0
        self.mask[:, :, h // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedCNN, self).forward(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, padding=3, use_masked=True, mask_type='B'):
        super(ResidualBlock, self).__init__()
        self.use_masked = use_masked
        self.conv1 = MaskedCNN(mask_type, in_channels, out_channels, kernel_size, padding=padding) if use_masked else nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = MaskedCNN(mask_type, out_channels, out_channels, kernel_size, padding=padding) if use_masked else nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.skip:
            residual = self.skip(residual)
        x += residual
        return self.relu(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # (B, H*W, C//8)
        key = self.key(x).view(batch_size, -1, H * W)  # (B, C//8, H*W)
        attention = torch.bmm(query, key)  # (B, H*W, H*W)
        attention = F.softmax(attention, dim=-1)
        value = self.value(x).view(batch_size, -1, H * W)  # (B, C, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, H*W)
        out = out.view(batch_size, C, H, W)
        out = self.gamma * out + x
        return out

class PixelCNN(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, kernel_size=7, num_layers=12, attention_every=4):
        super(PixelCNN, self).__init__()

        # Embedding layer for the latent codes
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        # Initial MaskedCNN layer with mask type 'A'
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
            MaskedCNN('A', embedding_dim, embedding_dim, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU()
        ))

        # Add layers and integrate Self-Attention
        for i in range(1, num_layers):
            self.layers.append(ResidualBlock(embedding_dim, embedding_dim))

            # Add a self-attention layer after every 'attention_every' residual layers
            # if i % attention_every == 0:
            #     self.layers.append(SelfAttentionBlock(embedding_dim))

        # Output layer for logits
        self.out_layer = nn.Sequential(
            nn.Conv2d(embedding_dim, 1024, 1),
            nn.ReLU(True),
            nn.Conv2d(1024, num_embeddings, 1)  # Predict logits over latent code indices
        )

    def forward(self, x):
        # Convert latent code indices to embeddings
        x = self.embedding(x).permute(0, 3, 1, 2)  # Convert to shape [batch_size, embedding_dim, height, width]
        for layer in self.layers:
            x = layer(x)
        return self.out_layer(x)  # Return logits over latent code indices