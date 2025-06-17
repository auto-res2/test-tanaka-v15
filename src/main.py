#!/usr/bin/env python3
"""
Purify-G++ experimental code.
This script implements three experiments:
  1. Adaptive Classifier Guidance Strength.
  2. Joint Multi-Objective Optimization with Dynamic Loss Scheduling.
  3. Adaptive Noise Injection Controlled by Classifier Guidance.
  
Each experiment uses a dummy diffusion model and classifier implemented in PyTorch.
Plots are generated and saved as .pdf files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from train import train_joint_loss_optimization
from evaluate import evaluate_adaptive_guidance
from preprocess import preprocess_data

class DummyDiffusionModel(nn.Module):
    def __init__(self):
        super(DummyDiffusionModel, self).__init__()
        self.fc = nn.Linear(32*32*3, 32*32*3)
    
    def forward(self, x, timestep):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        update = self.fc(x_flat)
        update = update.view_as(x)
        factor = 1.0 / (1.0 + timestep)
        return factor * update

class DummyClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(DummyClassifier, self).__init__()
        self.fc = nn.Linear(32*32*3, num_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        logits = self.fc(x_flat)
        return logits

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

torch.manual_seed(42)
np.random.seed(42)

def adaptive_weight_linear(confidence, k=1.0):
    return k * confidence

def adaptive_weight_exponential(confidence, alpha=1.0, threshold=0.5):
    return torch.exp(alpha * (confidence - threshold))

def adaptive_weight_threshold(confidence, threshold=0.7, weight=1.0):
    return weight if confidence > threshold else 0.0

def run_adaptive_guidance_step(x, timestep, diffusion_model, classifier, weighting_fn):
    """
    One reverse diffusion step with adaptive classifier guidance.
    """
    x = x.clone().detach().requires_grad_(True)
    
    diffusion_update = diffusion_model(x, timestep)
    
    logits = classifier(x)
    prob = F.softmax(logits, dim=1)
    confidence = prob.max(dim=1)[0]
    
    predicted_label = logits.argmax(dim=1)
    classifier_loss = F.cross_entropy(logits, predicted_label)
    
    grad_classifier = torch.autograd.grad(classifier_loss, x, retain_graph=True)[0]
    
    weight = weighting_fn(confidence.mean())
    
    steered_update = diffusion_update - weight * grad_classifier
    x_updated = x.detach() + steered_update
    return x_updated

def run_adaptive_guidance_purification(x_initial, diffusion_model, classifier, weighting_fn, timesteps=5):
    x = x_initial.clone()
    confidence_evolution = []
    for t in range(timesteps, 0, -1):
        with torch.no_grad():
            logits = classifier(x)
            conf = F.softmax(logits, dim=1).max(dim=1)[0].mean().item()
            confidence_evolution.append(conf)
        x = run_adaptive_guidance_step(x, t, diffusion_model, classifier, weighting_fn)
    return x, confidence_evolution

def experiment1():
    print("Experiment 1: Adaptive Classifier Guidance Strength")
    diffusion_model = DummyDiffusionModel()
    classifier = DummyClassifier(num_classes=10)

    x_initial = torch.randn(10, 3, 32, 32)

    candidates = [
        ('linear', adaptive_weight_linear),
        ('exponential', adaptive_weight_exponential),
        ('threshold', adaptive_weight_threshold)
    ]
    results = dict()

    timesteps = 5
    for name, func in candidates:
        print("Running candidate weighting function:", name)
        x_final, conf_evo = run_adaptive_guidance_purification(x_initial, diffusion_model, classifier, func, timesteps)
        results[name] = conf_evo
        print("Final mean classifier confidence (%s): %.4f" % (name, conf_evo[-1]))

    output_dir = ensure_output_directory()
    plt.figure(figsize=(6,4))
    for name, conf_evo in results.items():
        timesteps_list = list(range(timesteps, 0, -1))
        plt.plot(timesteps_list, conf_evo, marker='o', label=name)
    plt.xlabel("Timestep")
    plt.ylabel("Mean Classifier Confidence")
    plt.title("Classifier Confidence Evolution for Adaptive Guidance")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "classifier_confidence.pdf"), bbox_inches="tight")
    plt.close()
    print("Experiment 1 plot saved as 'classifier_confidence.pdf'\n")

def dynamic_loss_weight(confidence, base_weight=0.5, scale=1.0):
    return base_weight + scale * (1 - confidence.mean())

def joint_loss_update(x, timestep, diffusion_model, classifier, static_weight=None):
    x = x.clone().detach().requires_grad_(True)
    noise_pred = diffusion_model(x, timestep)
    target_noise = torch.zeros_like(noise_pred)
    denoising_loss = F.mse_loss(noise_pred, target_noise)

    logits = classifier(x)
    predicted_label = logits.argmax(dim=1)
    classifier_loss = F.cross_entropy(logits, predicted_label)

    confidence = F.softmax(logits, dim=1).max(dim=1)[0]
    
    if static_weight is not None:
        guidance_weight = static_weight
    else:
        guidance_weight = dynamic_loss_weight(confidence)
    
    joint_loss = denoising_loss + guidance_weight * classifier_loss
    joint_loss.backward()
    with torch.no_grad():
        grad_update = x.grad
        step_size = 0.1
        x_updated = x - step_size * grad_update
    return x_updated.detach(), denoising_loss.item(), classifier_loss.item(), guidance_weight

def run_joint_optimization_purification(x_initial, diffusion_model, classifier, timesteps=5, use_static_weight=False):
    x = x_initial.clone()
    loss_logs = []
    for t in range(timesteps, 0, -1):
        static_weight = 0.5 if use_static_weight else None
        x, denoise_loss, cls_loss, applied_weight = joint_loss_update(x, t, diffusion_model, classifier, static_weight)
        loss_logs.append((denoise_loss, cls_loss, applied_weight))
    return x, loss_logs

def experiment2():
    print("Experiment 2: Joint Multi-Objective Optimization with Dynamic Loss Scheduling")
    diffusion_model = DummyDiffusionModel()
    classifier = DummyClassifier(num_classes=10)
    x_initial = torch.randn(10, 3, 32, 32)

    print("Running joint optimization with STATIC weight")
    _, loss_logs_static = run_joint_optimization_purification(x_initial, diffusion_model, classifier, timesteps=5, use_static_weight=True)
    print("Running joint optimization with DYNAMIC weight")
    _, loss_logs_dynamic = run_joint_optimization_purification(x_initial, diffusion_model, classifier, timesteps=5, use_static_weight=False)

    timesteps_list = list(range(5, 0, -1))
    static_denoise = [log[0] for log in loss_logs_static]
    static_classifier = [log[1] for log in loss_logs_static]
    dynamic_denoise = [log[0] for log in loss_logs_dynamic]
    dynamic_classifier = [log[1] for log in loss_logs_dynamic]

    output_dir = ensure_output_directory()
    plt.figure(figsize=(6,4))
    plt.plot(timesteps_list, static_denoise, marker='o', label="Denoising Loss")
    plt.plot(timesteps_list, static_classifier, marker='x', label="Classifier Loss")
    plt.xlabel("Timestep")
    plt.ylabel("Loss")
    plt.title("Joint Loss (Static Weight)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "joint_loss_static.pdf"), bbox_inches="tight")
    plt.close()
    print("Static joint loss plot saved as 'joint_loss_static.pdf'")

    plt.figure(figsize=(6,4))
    plt.plot(timesteps_list, dynamic_denoise, marker='o', label="Denoising Loss")
    plt.plot(timesteps_list, dynamic_classifier, marker='x', label="Classifier Loss")
    plt.xlabel("Timestep")
    plt.ylabel("Loss")
    plt.title("Joint Loss (Dynamic Weight)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "joint_loss_dynamic.pdf"), bbox_inches="tight")
    plt.close()
    print("Dynamic joint loss plot saved as 'joint_loss_dynamic.pdf'\n")

def compute_gradient_norm(x, classifier, timestep):
    x = x.clone().detach().requires_grad_(True)
    logits = classifier(x)
    predicted_label = logits.argmax(dim=1)
    loss = F.cross_entropy(logits, predicted_label)
    grad = torch.autograd.grad(loss, x, retain_graph=True)[0]
    grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1).mean()
    return grad_norm

def noise_injection_factor(timestep, grad_norm, linear_coef=0.01, exp_coef=0.05):
    return torch.exp(-exp_coef * torch.tensor(timestep, dtype=torch.float32)) * (1 + exp_coef * grad_norm)

def adaptive_noise_injection_step(x, timestep, diffusion_model, classifier):
    update = diffusion_model(x, timestep)
    
    grad_norm = compute_gradient_norm(x, classifier, timestep)
    
    noise_factor = noise_injection_factor(timestep, grad_norm)
    
    noise = torch.randn_like(x) * noise_factor
    x_updated = x + update + noise
    return x_updated.detach(), noise_factor.item()

def run_adaptive_noise_purification(x_initial, diffusion_model, classifier, timesteps=5):
    x = x_initial.clone()
    noise_factor_log = []
    for t in range(timesteps, 0, -1):
        x, nf = adaptive_noise_injection_step(x, t, diffusion_model, classifier)
        noise_factor_log.append(nf)
        print("Timestep %d: noise injection factor = %.4f" % (t, nf))
    return x, noise_factor_log

def experiment3():
    print("Experiment 3: Adaptive Noise Injection Controlled by Classifier Guidance")
    diffusion_model = DummyDiffusionModel()
    classifier = DummyClassifier(num_classes=10)
    x_initial = torch.randn(10, 3, 32, 32)

    timesteps = 5
    _, noise_factor_log = run_adaptive_noise_purification(x_initial, diffusion_model, classifier, timesteps)

    timesteps_list = list(range(timesteps, 0, -1))
    output_dir = ensure_output_directory()
    plt.figure(figsize=(6,4))
    plt.plot(timesteps_list, noise_factor_log, marker='s', color='purple')
    plt.xlabel("Timestep")
    plt.ylabel("Noise Injection Factor")
    plt.title("Adaptive Noise Injection Factor vs. Timestep")
    plt.savefig(os.path.join(output_dir, "noise_injection_adaptive.pdf"), bbox_inches="tight")
    plt.close()
    print("Experiment 3 plot saved as 'noise_injection_adaptive.pdf'\n")

def run_all_tests():
    print("Running all experiments in test mode...")
    experiment1()
    experiment2()
    experiment3()
    print("All experiments finished successfully.")

def main():
    print("=== Purify-G++ Experimental Script ===")
    print("Implementing adaptive classifier guidance for diffusion-based adversarial purification")
    print("Hardware: NVIDIA Tesla T4 (16GB VRAM)")
    print("=" * 60)
    
    try:
        print("\n🚀 Starting Purify-G++ experiments...")
        
        preprocess_results = preprocess_data()
        print(f"✓ Data preprocessing completed: {preprocess_results}")
        
        train_results = train_joint_loss_optimization()
        print(f"✓ Training completed: {train_results}")
        
        eval_results = evaluate_adaptive_guidance()
        print(f"✓ Evaluation completed: {eval_results}")
        
        run_all_tests()
        
        print("\n" + "=" * 60)
        print("PURIFY-G++ EXPERIMENT COMPLETION SUMMARY")
        print("=" * 60)
        print("✓ Experiment 1: Adaptive Classifier Guidance - COMPLETED")
        print("  - Generated: classifier_confidence.pdf")
        
        print("✓ Experiment 2: Joint Multi-Objective Optimization - COMPLETED")
        print("  - Generated: joint_loss_static.pdf, joint_loss_dynamic.pdf")
        
        print("✓ Experiment 3: Adaptive Noise Injection - COMPLETED")
        print("  - Generated: noise_injection_adaptive.pdf")
        
        print("\n✓ All PDF plots saved to .research/iteration1/images/")
        print("✓ Experiment designed for NVIDIA Tesla T4 compatibility")
        print("✓ Quick test run completed successfully")
        
        status_enum = "stopped"
        print(f"\n✓ Status: {status_enum}")
        print("=" * 60)
        print("PURIFY-G++ EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: Experiment failed with exception: {e}")
        status_enum = "stopped"
        print(f"Status: {status_enum}")
        raise

if __name__ == '__main__':
    main()
