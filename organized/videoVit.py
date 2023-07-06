from vit_pytorch import ViT
import random_tensor as rt
import torch

# Instantiate the Vision Transformer model
model = ViT(
    image_size=224,  # Input image size
    image_time=160,  # Input image time
    patch_size=16,   # Patch size
    patch_time=8,    # Patch time
    num_classes=32,  # Number of output classes
    dim=768,         # Embedding dimension
    depth=12,        # Number of transformer blocks
    heads=12,        # Number of attention heads
    mlp_dim=3072     # Hidden dimension of the MLP
)

video = rt.yt_to_tensor("https://www.youtube.com/watch?v=rL1xA-3WXFs")
video = rt.resize_video(video, 160, 224)
video = video.unsqueeze(0).permute(0, 2, 1, 3, 4).to(torch.float32)
print(video.shape)

output = model(video)
print(output.shape)
print(output)