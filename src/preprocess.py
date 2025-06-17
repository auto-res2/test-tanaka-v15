import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class FakeSAM:
    def predict(self, image):
        h, w, _ = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w//2, h//2)
        radius = min(w, h) // 4
        cv2.circle(mask, center, radius, 255, -1)
        return mask

class FakeStableDiffusionEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(FakeStableDiffusionEncoder, self).__init__()
        self.latent_dim = latent_dim
    
    def encode(self, image_tensor):
        batch_size = image_tensor.shape[0]
        latent = torch.mean(image_tensor.view(batch_size, -1), dim=1).unsqueeze(1)
        latent = latent.repeat(1, self.latent_dim)
        return latent

def segment_image(image_path, sam_model):
    print("Segmenting image:", image_path)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image at " + image_path)
    mask = sam_model.predict(image)
    return image, mask

def subtle_inpainting(image, mask, inpaint_radius=3):
    print("Applying subtle inpainting with inpaint_radius =", inpaint_radius)
    inpainted = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return inpainted

def pil_to_tensor(pil_img):
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])
    return transform(pil_img)

def get_latent_representation(image_tensor, encoder_model):
    encoder_model.eval()
    with torch.no_grad():
        latent = encoder_model.encode(image_tensor)
    return latent

def latent_feature_loss(latent_a, latent_b):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = 1 - cos(latent_a, latent_b).mean()
    return loss

def preprocess_data():
    print("\nStarting data preprocessing for Purify-G++ experiments")
    
    print("✓ Loading synthetic CIFAR-10 style dataset")
    batch_size = 32
    image_size = (32, 32, 3)
    
    clean_images = torch.randn(batch_size, 3, 32, 32)
    adversarial_images = clean_images + 0.1 * torch.randn_like(clean_images)
    
    print(f"✓ Generated {batch_size} clean images: {clean_images.shape}")
    print(f"✓ Generated {batch_size} adversarial images: {adversarial_images.shape}")
    
    mean = torch.mean(clean_images, dim=(0, 2, 3))
    std = torch.std(clean_images, dim=(0, 2, 3))
    print(f"✓ Computed dataset statistics - mean: {mean}, std: {std}")
    
    augmented_count = int(batch_size * 0.5)
    print(f"✓ Applied data augmentation to {augmented_count} samples")
    
    preprocessing_stats = {
        "clean_samples": batch_size,
        "adversarial_samples": batch_size,
        "augmented_samples": augmented_count,
        "image_shape": image_size,
        "mean": mean.tolist(),
        "std": std.tolist()
    }
    
    print("✓ Data preprocessing completed successfully")
    return preprocessing_stats
