"""
Utility functions for FL-NILM
"""
import os
import random
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def create_directories(config):
    """Create necessary directories"""
    directories = [
        config.PLAID_DATA_PATH,
        config.PROCESSED_DATA_PATH,
        config.SPLITS_PATH,
        config.MODEL_SAVE_PATH,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def load_client_splits(config):
    """Load client data splits"""
    splits_file = os.path.join(config.SPLITS_PATH, 'client_splits.pkl')
    
    if not os.path.exists(splits_file):
        raise FileNotFoundError(
            f"Client splits not found at {splits_file}. "
            "Please run preprocessing first."
        )
    
    with open(splits_file, 'rb') as f:
        client_splits = pickle.load(f)
    
    return client_splits


def plot_training_history(history, save_path):
    """
    Plot training history
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training Loss
    if history['train_loss']:
        axes[0, 0].plot(history['round'], history['train_loss'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Training Accuracy
    if history['train_acc']:
        axes[0, 1].plot(history['round'], history['train_acc'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Test Loss
    if history['test_loss']:
        test_rounds = [history['round'][i] for i in range(len(history['test_loss']))]
        axes[1, 0].plot(test_rounds, history['test_loss'], 'r-', 
                       linewidth=2, marker='o')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Test Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Test Accuracy
    if history['test_acc']:
        test_rounds = [history['round'][i] for i in range(len(history['test_acc']))]
        axes[1, 1].plot(test_rounds, history['test_acc'], 'm-', 
                       linewidth=2, marker='o')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Test Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('FL-NILM Training Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def save_results(history, config, save_dir):
    """Save training results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save history
    history_path = os.path.join(save_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    # Save config
    config_path = os.path.join(save_dir, 'config.pkl')
    config_dict = {
        'model_name': config.MODEL_NAME,
        'aggregation': config.AGGREGATION_METHOD,
        'num_clients': config.NUM_CLIENTS,
        'num_rounds': config.NUM_ROUNDS,
        'local_epochs': config.LOCAL_EPOCHS,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'use_dp': config.USE_DP,
    }
    
    with open(config_path, 'wb') as f:
        pickle.dump(config_dict, f)
    
    # Plot results
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plot_training_history(history, plot_path)
    
    print(f"\nResults saved to {save_dir}")


def print_config(config):
    """Print configuration"""
    print("\n" + "="*60)
    print("FL-NILM Configuration")
    print("="*60)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Aggregation: {config.AGGREGATION_METHOD}")
    print(f"Clients: {config.NUM_CLIENTS}")
    print(f"Rounds: {config.NUM_ROUNDS}")
    print(f"Local Epochs: {config.LOCAL_EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Use DP: {config.USE_DP}")
    if config.USE_DP:
        print(f"  DP Epsilon: {config.DP_EPSILON}")
    print(f"Device: {config.DEVICE}")
    print(f"Seed: {config.SEED}")
    print(f"Use WandB: {config.USE_WANDB}")
    print("="*60)
