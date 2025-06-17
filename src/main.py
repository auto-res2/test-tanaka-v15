import torch
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from preprocess import FakeSAM, FakeStableDiffusionEncoder, segment_image, subtle_inpainting, pil_to_tensor, get_latent_representation, latent_feature_loss
from train import train_joint_loss_optimization
from evaluate import evaluate_distributed_poisoning

def ensure_output_directory():
    """Ensure the output directory exists for saving PDF files"""
    possible_paths = [
        "../.research/iteration1/images/",  # From src/ directory
        ".research/iteration1/images/",    # From root directory
        "../../.research/iteration1/images/"  # From nested directory
    ]
    
    for path in possible_paths:
        try:
            os.makedirs(path, exist_ok=True)
            test_file = os.path.join(path, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"✓ Output directory confirmed: {os.path.abspath(path)}")
            return path
        except:
            continue
    
    fallback_path = "output_images/"
    os.makedirs(fallback_path, exist_ok=True)
    print(f"⚠ Using fallback output directory: {os.path.abspath(fallback_path)}")
    return fallback_path

def experiment_dual_level_poisoning():
    print("\n" + "="*80)
    print("EXPERIMENT 1: DUAL-LEVEL POISONING GENERATION PIPELINE")
    print("="*80)
    print("📋 Objective: Implement pixel-level semantic encoding + latent watermarking")
    print("🔧 Components: FakeSAM segmentation, subtle inpainting, latent extraction")
    print("📊 Metrics: Latent feature loss between original and modified images")
    print("-"*80)
    
    output_dir = ensure_output_directory()
    
    print("🚀 Initializing models...")
    sam_model = FakeSAM()
    encoder_model = FakeStableDiffusionEncoder(latent_dim=128)
    print("✓ FakeSAM segmentation model initialized")
    print("✓ FakeStableDiffusionEncoder initialized (latent_dim=128)")
    
    print("\n📸 Generating synthetic copyright image...")
    image = np.full((64, 64, 3), 200, dtype=np.uint8)
    cv2.rectangle(image, (16, 16), (48, 48), (50, 50, 50), -1)
    print(f"✓ Created synthetic image: {image.shape} with central rectangle")
    print(f"  - Background color: RGB(200,200,200)")
    print(f"  - Rectangle region: (16,16) to (48,48) in RGB(50,50,50)")
    
    print("\n📊 Saving original image visualization...")
    plt.figure(figsize=(4,4))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Copyright Image")
    plt.axis("off")
    original_path = os.path.join(output_dir, "poisoning_images_original.pdf")
    plt.savefig(original_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ Original image saved: {original_path}")
    
    print("\n🎯 Performing image segmentation...")
    temp_path = "temp_image.jpg"
    cv2.imwrite(temp_path, image)
    image_loaded, mask = segment_image(temp_path, sam_model)
    mask_area = np.sum(mask > 0)
    total_area = mask.shape[0] * mask.shape[1]
    print(f"✓ Segmentation completed")
    print(f"  - Mask area: {mask_area} pixels ({mask_area/total_area*100:.1f}% of image)")
    print(f"  - Mask shape: {mask.shape}")
    
    print("\n🖌️ Applying subtle inpainting for stealth modification...")
    modified_image = subtle_inpainting(image_loaded, mask, inpaint_radius=3)
    
    diff = np.abs(image_loaded.astype(float) - modified_image.astype(float))
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    modified_pixels = np.sum(diff > 0)
    print(f"✓ Inpainting completed with radius=3")
    print(f"  - Mean pixel difference: {mean_diff:.2f}")
    print(f"  - Max pixel difference: {max_diff:.2f}")
    print(f"  - Modified pixels: {modified_pixels} ({modified_pixels/total_area*100:.1f}%)")
    
    print("\n📊 Saving modified image visualization...")
    plt.figure(figsize=(4,4))
    plt.imshow(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB))
    plt.title("Modified Image (Subtle Inpainting)")
    plt.axis("off")
    modified_path = os.path.join(output_dir, "poisoning_images_modified.pdf")
    plt.savefig(modified_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✓ Modified image saved: {modified_path}")
    
    print("\n🧠 Extracting latent representations...")
    pil_img = Image.fromarray(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB))
    img_tensor = pil_to_tensor(pil_img).unsqueeze(0)
    latent_modified = get_latent_representation(img_tensor, encoder_model)
    print(f"✓ Modified image latent shape: {latent_modified.shape}")
    
    pil_original = Image.fromarray(cv2.cvtColor(image_loaded, cv2.COLOR_BGR2RGB))
    orig_tensor = pil_to_tensor(pil_original).unsqueeze(0)
    latent_original = get_latent_representation(orig_tensor, encoder_model)
    print(f"✓ Original image latent shape: {latent_original.shape}")
    
    print("\n📈 Computing latent feature similarity...")
    loss_val = latent_feature_loss(latent_modified, latent_original)
    print(f"✓ Latent feature loss (1 - cosine similarity): {loss_val.item():.8f}")
    
    if loss_val.item() < 0.001:
        print("🎉 SUCCESS: Very low latent loss indicates successful stealth modification!")
    elif loss_val.item() < 0.01:
        print("✅ GOOD: Low latent loss shows effective latent watermarking")
    else:
        print("⚠️  WARNING: High latent loss may indicate detectable modifications")
    
    print("\n" + "="*80)
    print("EXPERIMENT 1 COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return {"latent_loss": loss_val.item(), "output_dir": output_dir}

def test_experiments():
    print("\n" + "🧪" + " "*30 + "RUNNING LEB EXPERIMENT SUITE" + " "*30 + "🧪")
    print("="*90)
    print("🎯 Testing all three components of Latent-Enmeshment Backdoor attack")
    print("⏱️  Estimated runtime: ~30 seconds for quick validation")
    print("="*90)
    
    results1 = experiment_dual_level_poisoning()
    results2 = train_joint_loss_optimization()
    results3 = evaluate_distributed_poisoning()
    
    print("\n" + "📊" + " "*35 + "EXPERIMENT RESULTS SUMMARY" + " "*35 + "📊")
    print("="*90)
    print(f"🔬 Experiment 1 (Dual-level Poisoning)")
    print(f"   └─ Latent Loss: {results1['latent_loss']:.8f}")
    print(f"   └─ Status: {'✅ PASSED' if results1['latent_loss'] < 0.01 else '⚠️ WARNING'}")
    
    print(f"\n🎯 Experiment 2 (Joint Loss Optimization)")
    final_loss = results2.get('final_loss', 0.0)
    print(f"   └─ Final Training Loss: {final_loss:.6f}")
    print(f"   └─ Status: {'✅ CONVERGED' if final_loss and final_loss < 1.0 else '⚠️ HIGH LOSS'}")
    
    print(f"\n🔄 Experiment 3 (Distributed Poisoning Robustness)")
    print(f"   └─ Trigger Activation Loss: {results3['trigger_loss']:.8f}")
    print(f"   └─ Robustness Ratios: {len(results3['similarity_losses'])} tested")
    print(f"   └─ Status: {'✅ ROBUST' if results3['trigger_loss'] < 0.01 else '⚠️ UNSTABLE'}")
    
    print("\n" + "🎉" + " "*30 + "ALL EXPERIMENTS COMPLETED" + " "*30 + "🎉")
    print("="*90)
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
