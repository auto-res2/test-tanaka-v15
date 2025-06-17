import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from preprocess import FakeStableDiffusionEncoder, get_latent_representation, latent_feature_loss

class FakeDiffusionModel(nn.Module):
    def __init__(self):
        super(FakeDiffusionModel, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class FakeCLIPModule:
    def __init__(self, embed_dim=512):
        self.embed_dim = embed_dim
    
    def encode_image(self, image_tensor):
        batch_size = image_tensor.shape[0]
        embed = torch.randn(batch_size, self.embed_dim)
        return nn.functional.normalize(embed, dim=1)
    
    def encode_text(self, text_list):
        batch_size = len(text_list)
        embed = torch.randn(batch_size, self.embed_dim)
        return nn.functional.normalize(embed, dim=1)

def compute_clip_loss(image_tensor, text_prompt, clip_module):
    image_embed = clip_module.encode_image(image_tensor)
    text_embed = clip_module.encode_text([text_prompt])
    text_embed = text_embed.expand_as(image_embed)
    sim = nn.functional.cosine_similarity(image_embed, text_embed, dim=1)
    loss = 1 - sim.mean()
    return loss

def train_joint_loss_optimization():
    print("\nStarting Joint Loss Optimization Training")
    
    diffusion_model = FakeDiffusionModel()
    encoder_model = FakeStableDiffusionEncoder(latent_dim=128)
    clip_module = FakeCLIPModule(embed_dim=512)
    
    batch_size = 4
    num_batches = 3
    
    optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-3)
    
    training_losses = []
    
    for epoch in range(2):
        print("Epoch:", epoch)
        for batch in range(num_batches):
            clean_images = torch.rand(batch_size, 3, 64, 64)
            poisoned_images = torch.rand(batch_size, 3, 64, 64)
            watermark_refs = torch.rand(batch_size, 3, 64, 64)
            
            optimizer.zero_grad()
            
            reconstructed_clean = diffusion_model(clean_images)
            diffusion_loss = nn.functional.mse_loss(reconstructed_clean, clean_images)
            
            reconstructed_poison = diffusion_model(poisoned_images)
            
            text_prompt = "special trigger description"
            semantic_loss = compute_clip_loss(reconstructed_poison, text_prompt, clip_module)
            
            latent_poison = get_latent_representation(poisoned_images, encoder_model)
            latent_ref = get_latent_representation(watermark_refs, encoder_model)
            watermark_loss = latent_feature_loss(latent_poison, latent_ref)
            
            total_loss = diffusion_loss + 0.5 * semantic_loss + 0.3 * watermark_loss
            total_loss.backward()
            optimizer.step()
            
            training_losses.append(total_loss.item())
            print("  Batch", batch, "Total Loss:", total_loss.item())
    
    plt.figure(figsize=(5,3))
    plt.plot(training_losses, marker='o', label="Total Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Batch iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("../.research/iteration1/images/training_loss_baseline.pdf", bbox_inches="tight")
    plt.close()

    print("Joint loss optimization training completed. Training loss plot saved.")
    return {"final_loss": training_losses[-1] if training_losses else None}
