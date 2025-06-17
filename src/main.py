import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from preprocess import FakeSAM, FakeStableDiffusionEncoder, segment_image, subtle_inpainting, pil_to_tensor, get_latent_representation, latent_feature_loss
from train import train_joint_loss_optimization
from evaluate import evaluate_distributed_poisoning

def experiment_dual_level_poisoning():
    print("\nStarting Experiment 1: Dual-Level Poisoning Generation Pipeline")
    
    sam_model = FakeSAM()
    encoder_model = FakeStableDiffusionEncoder(latent_dim=128)
    
    image = np.full((64, 64, 3), 200, dtype=np.uint8)
    cv2.rectangle(image, (16, 16), (48, 48), (50, 50, 50), -1)
    
    plt.figure(figsize=(4,4))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    plt.savefig("../.research/iteration1/images/poisoning_images_original.pdf", bbox_inches="tight")
    plt.close()
    
    temp_path = "temp_image.jpg"
    cv2.imwrite(temp_path, image)
    image_loaded, mask = segment_image(temp_path, sam_model)
    
    modified_image = subtle_inpainting(image_loaded, mask, inpaint_radius=3)
    
    plt.figure(figsize=(4,4))
    plt.imshow(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB))
    plt.title("Modified Image (Inpainted)")
    plt.axis("off")
    plt.savefig("../.research/iteration1/images/poisoning_images_modified.pdf", bbox_inches="tight")
    plt.close()
    
    pil_img = Image.fromarray(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB))
    img_tensor = pil_to_tensor(pil_img).unsqueeze(0)
    latent_modified = get_latent_representation(img_tensor, encoder_model)
    
    pil_original = Image.fromarray(cv2.cvtColor(image_loaded, cv2.COLOR_BGR2RGB))
    orig_tensor = pil_to_tensor(pil_original).unsqueeze(0)
    latent_original = get_latent_representation(orig_tensor, encoder_model)
    
    loss_val = latent_feature_loss(latent_modified, latent_original)
    print("Latent feature loss between modified and original images:", loss_val.item())
    
    return {"latent_loss": loss_val.item()}

def test_experiments():
    print("\nRunning quick tests for experiments...")
    results1 = experiment_dual_level_poisoning()
    results2 = train_joint_loss_optimization()
    results3 = evaluate_distributed_poisoning()
    print("\nTest results summary:")
    print("Experiment 1 (Dual-level Poisoning) Latent Loss:", results1["latent_loss"])
    print("Experiment 2 (Joint Loss Optimization) Final Loss:", results2["final_loss"])
    print("Experiment 3 (Distributed Poisoning) Trigger Loss:", results3["trigger_loss"])
    print("\nAll tests completed successfully.")
    return results1, results2, results3

def main():
    print("=== Latent-Enmeshment Backdoor (LEB) Experiment ===")
    print("Implementing novel backdoor attack on diffusion models")
    print("Hardware: NVIDIA Tesla T4 (16GB VRAM)")
    print("=" * 60)
    
    try:
        results1, results2, results3 = test_experiments()
        
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETION SUMMARY")
        print("=" * 60)
        print("✓ Experiment 1: Dual-Level Poisoning - COMPLETED")
        print(f"  - Latent Loss: {results1['latent_loss']:.6f}")
        print("  - Generated: poisoning_images_original.pdf, poisoning_images_modified.pdf")
        
        print("✓ Experiment 2: Joint Loss Optimization - COMPLETED")
        print(f"  - Final Training Loss: {results2['final_loss']:.6f}")
        print("  - Generated: training_loss_baseline.pdf")
        
        print("✓ Experiment 3: Distributed Poisoning Robustness - COMPLETED")
        print(f"  - Trigger Activation Loss: {results3['trigger_loss']:.6f}")
        print("  - Generated: inference_latency_distributed.pdf")
        
        print("\n✓ All PDF plots saved to .research/iteration1/images/")
        print("✓ Experiment designed for NVIDIA Tesla T4 compatibility")
        print("✓ Quick test run completed successfully")
        
        status_enum = "stopped"
        print(f"\n✓ Status: {status_enum}")
        print("=" * 60)
        print("LEB EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: Experiment failed with exception: {e}")
        status_enum = "stopped"
        print(f"Status: {status_enum}")
        raise

if __name__ == '__main__':
    main()
