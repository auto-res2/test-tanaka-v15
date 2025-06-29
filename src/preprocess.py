import torch
import numpy as np
import random

def set_seed(seed):
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_noise_vectors(num_samples=8, shape=(3, 32, 32)):
    """Generate fixed noise vectors for RASID experiments."""
    set_seed(42)
    noise_vectors = [torch.randn(*shape) for _ in range(num_samples)]
    return noise_vectors

def preprocess_data():
    print("\nStarting data preprocessing for RASID experiments")
    
    print("✓ Generating fixed noise vectors for reproducibility testing")
    num_samples = 8
    noise_vectors = generate_noise_vectors(num_samples)
    
    print(f"✓ Generated {num_samples} fixed noise vectors: shape {noise_vectors[0].shape}")
    
    print("✓ Setting up reproducibility profiling parameters")
    reproducibility_params = {
        "num_runs": 5,
        "timesteps": 5,
        "noise_samples": num_samples
    }
    
    print("✓ Initializing RASID preprocessing parameters")
    rasid_params = {
        "lambda_values": [0.0, 0.1, 0.5, 1.0],
        "training_epochs": 3,
        "batch_size": 3
    }
    
    preprocessing_stats = {
        "noise_samples": num_samples,
        "noise_shape": list(noise_vectors[0].shape),
        "reproducibility_params": reproducibility_params,
        "rasid_params": rasid_params
    }
    
    print("✓ RASID data preprocessing completed successfully")
    return preprocessing_stats
