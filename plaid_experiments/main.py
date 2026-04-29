"""
Main entry point for FL-NILM
Supports multiple models and aggregation methods with WandB logging
"""
import os
import argparse
import pickle
import torch

from config import Config
from models import create_model, count_parameters
from client import ClientManager
from server import FLServer, FederatedTrainer
from utils import (
    set_seed, 
    get_device, 
    create_directories, 
    load_client_splits,
    save_results,
    print_config
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FL-NILM: Federated Learning for NILM')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate'],
                       help='Mode: train or evaluate')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='cnn',
                       choices=Config.AVAILABLE_MODELS,
                       help='Model architecture')
    
    # Aggregation configuration
    parser.add_argument('--aggregation', type=str, default='fedavg',
                       choices=Config.AVAILABLE_AGGREGATIONS,
                       help='Aggregation method')
    
    # FL parameters
    parser.add_argument('--num_clients', type=int, default=100,
                       help='Number of FL clients')
    parser.add_argument('--num_rounds', type=int, default=100,
                       help='Number of FL rounds')
    parser.add_argument('--kd_start_round', type=int, default=100,
                       help='Number of FL rounds')
    parser.add_argument('--local_epochs', type=int, default=2,
                       help='Local epochs per round')
    parser.add_argument('--client_fraction', type=float, default=1.0,
                       help='Fraction of clients per round')
    parser.add_argument('--client_selection_mode', type=str, default='fixed',
                       help='fixed client or random clients')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--Tlr', type=float, default=0.001,
                       help='Teacher Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    
    # Aggregation-specific parameters
    parser.add_argument('--fedprox_mu', type=float, default=0.01,
                       help='FedProx proximal term coefficient')
    parser.add_argument('--scaffold_lr', type=float, default=1.0,
                       help='SCAFFOLD learning rate')
    
    # Differential Privacy
    parser.add_argument('--use_dp', action='store_true',
                       help='Enable differential privacy')
    parser.add_argument('--dp_epsilon', type=float, default=100,
                       help='DP privacy budget')
    parser.add_argument('--dp_delta', type=float, default=1e-2,
                       help='DP delta parameter')
    parser.add_argument('--dp_clip_norm', type=float, default=5,
                       help='DP clip norm parameter')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='FL-NILM',
                       help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='',
                       help='WandB entity (your wandb username/team)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    
    # Evaluation
    parser.add_argument('--eval_interval', type=int, default=10,
                       help='Evaluation interval (rounds)')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='Checkpoint save interval (rounds)')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of data loader workers')
    
    return parser.parse_args()


def update_config_from_args(args):
    """Update Config with command line arguments"""
    # Model
    Config.MODEL_NAME = args.model
    
    # Aggregation
    Config.AGGREGATION_METHOD = args.aggregation
    Config.FEDPROX_MU = args.fedprox_mu
    Config.SCAFFOLD_LR = args.scaffold_lr
    
    # FL parameters
    Config.NUM_CLIENTS = args.num_clients
    Config.NUM_ROUNDS = args.num_rounds
    Config.KD_START_ROUND = args.kd_start_round
    Config.LOCAL_EPOCHS = args.local_epochs
    Config.CLIENT_FRACTION = args.client_fraction
    Config.CLIENT_SELECTION_MODE = args.client_selection_mode
    
    # Training
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.lr
    Config.TEACHER_LEARNING_RATE = args.Tlr
    Config.WEIGHT_DECAY = args.weight_decay
    
    # DP
    Config.USE_DP = args.use_dp
    Config.DP_EPSILON = args.dp_epsilon
    Config.DP_DELTA = args.dp_delta
    Config.DP_CLIP_NORM = args.dp_clip_norm
    
    # Logging
    Config.USE_WANDB = args.use_wandb
    Config.WANDB_PROJECT = args.wandb_project
    Config.WANDB_ENTITY = args.wandb_entity
    
    # Evaluation
    Config.EVAL_INTERVAL = args.eval_interval
    Config.SAVE_INTERVAL = args.save_interval
    
    # Other
    Config.SEED = args.seed
    Config.NUM_WORKERS = args.num_workers


def initialize_wandb(config, experiment_name=None):
    """Initialize Weights & Biases"""
    if not config.USE_WANDB:
        return None
    
    try:
        import wandb
        
        # Generate experiment name if not provided
        if experiment_name is None:
            experiment_name = config.get_experiment_name()
        tags = [
            config.MODEL_NAME,                          # Model: 'cnn', 'resnet18', etc.
            config.AGGREGATION_METHOD,                  # Aggregation: 'fedavg', 'perfednilm', etc.
            f'epochs-{config.LOCAL_EPOCHS}',            # Local epochs per round
            f'rounds-{config.NUM_ROUNDS}',              # Total federated rounds
            f'mode-{config.CLIENT_SELECTION_MODE}',
            f'fraction-{config.CLIENT_FRACTION}'
        ]
        if config.USE_DP:
            tags.append('dp')
        
        # Initialize wandb
        run = wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name=experiment_name,
            tags=tags,
            config={
                'model': config.MODEL_NAME,
                'aggregation': config.AGGREGATION_METHOD,
                'num_clients': config.NUM_CLIENTS,
                'num_rounds': config.NUM_ROUNDS,
                'local_epochs': config.LOCAL_EPOCHS,
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE,
                'weight_decay': config.WEIGHT_DECAY,
                'use_dp': config.USE_DP,
                'dp_epsilon': config.DP_EPSILON if config.USE_DP else None,
                'fedprox_mu': config.FEDPROX_MU if config.AGGREGATION_METHOD == 'fedprox' else None,
                'seed': config.SEED,
            }
        )
        
        print(f"\nWandB initialized: {experiment_name}")
        print(f"View at: {run.get_url()}")
        
        return run
        
    except ImportError:
        print("\nWarning: wandb not installed. Install with: pip install wandb")
        print("Continuing without WandB logging...")
        return None


def train(args):
    """Main training function"""
    
    # Update config
    update_config_from_args(args)
    
    # Set seed
    set_seed(Config.SEED)
    
    # Set device
    Config.DEVICE = get_device()
    
    # Create directories
    create_directories(Config)
    
    # Print configuration
    print_config(Config)
    
    # Load client splits
    # Load client splits
    print("\nLoading client data splits...")
    client_splits = load_client_splits(Config)
    if Config.CLIENT_SELECTION_MODE=='fixed':
        num_selected = max(1, int(len(client_splits) * Config.CLIENT_FRACTION))
        client_splits = client_splits[:num_selected]
    print(f"Loaded {len(client_splits)} client splits")
    
    # Determine the number of appliance classes.
    # Preferred: load from label_map.pkl produced by preprocess.py.
    label_map_path = os.path.join(Config.PROCESSED_DATA_PATH, 'label_map.pkl')
    class_names = None

    if os.path.exists(label_map_path):
        with open(label_map_path, 'rb') as f:
            label_map = pickle.load(f)
        num_classes = len(label_map)
        # label_map is {appliance_type_str: int_label}; derive ordered class names.
        class_names = [None] * num_classes
        for name, idx in label_map.items():
            class_names[idx] = name
        print(f"Loaded label map. Total classes: {num_classes}")
        print(f"Classes: {class_names}")
    else:
        # Fallback: scan all client splits to derive the maximum label.
        print("Warning: label_map.pkl not found. Scanning all clients...")
        max_label = 0
        for client_data in client_splits:
            for s in client_data['train'] + client_data['test']:
                if s['label'] > max_label:
                    max_label = s['label']
        num_classes = max_label + 1

    Config.update_num_classes(num_classes)

    # Route checkpoints into an experiment-specific subdirectory so that
    # concurrent / sequential runs never overwrite each other's weights.
    experiment_name = args.experiment_name or Config.get_experiment_name()
    Config.MODEL_SAVE_PATH = os.path.join(Config.MODEL_SAVE_PATH, experiment_name)
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)

    # Initialize WandB
    wandb_run = initialize_wandb(Config, args.experiment_name)
    
    # Create model factory function
    def model_fn():
        model_config = Config.MODEL_CONFIGS.get(Config.MODEL_NAME, {})
        return create_model(Config.MODEL_NAME, Config.NUM_CLASSES, model_config)
    
    # Initialize clients
    client_manager = ClientManager(client_splits, model_fn, Config)
    
    # Initialize server
    global_model = model_fn()
    print(f"\nGlobal model parameters: {count_parameters(global_model):,}")
    
    server = FLServer(global_model, Config, wandb_run, class_names=class_names)
    
    # Create trainer
    trainer = FederatedTrainer(server, client_manager, Config, wandb_run)
    
    # Train
    history = trainer.train()
    
    # Save results into the same experiment-specific directory as the checkpoints.
    save_results(history, Config, Config.MODEL_SAVE_PATH)
    
    # Finish wandb
    if wandb_run:
        wandb_run.finish()
    
    print("\nTraining completed successfully!")


def evaluate(args):
    """Evaluation function"""
    
    # Update config
    update_config_from_args(args)
    
    # Set device
    Config.DEVICE = get_device()
    
    print("\nEvaluation mode:")
    print("Loading trained model and evaluating on test set...")
    
    # Load client splits
    client_splits = load_client_splits(Config)
    
    # Get number of classes
    if client_splits:
        sample_labels = [s['label'] for s in client_splits[0]['train']]
        num_classes = len(set(sample_labels))
        Config.update_num_classes(num_classes)
    
    # Create model
    model_config = Config.MODEL_CONFIGS.get(Config.MODEL_NAME, {})
    model = create_model(Config.MODEL_NAME, Config.NUM_CLASSES, model_config)
    
    # Load checkpoint
    experiment_name = args.experiment_name or Config.get_experiment_name()
    checkpoint_path = os.path.join(Config.MODEL_SAVE_PATH, experiment_name, 'global_model.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {checkpoint_path}")
    
    # Initialize clients
    def model_fn():
        return create_model(Config.MODEL_NAME, Config.NUM_CLASSES, model_config)
    
    client_manager = ClientManager(client_splits, model_fn, Config)
    
    # Initialize server for evaluation
    server = FLServer(model, Config)
    
    # Evaluate
    print("\nEvaluating on all clients...")
    metrics = server.evaluate(client_manager.get_all_clients())
    
    print("\nEvaluation Results:")
    print(f"Average Test Loss: {metrics['test_loss']:.4f}")
    print(f"Average Test Accuracy: {metrics['test_acc']:.2f}%")
    
    print("\nPer-client accuracies:")
    for i, acc in enumerate(metrics['client_accs']):
        print(f"  Client {i}: {acc:.2f}%")


def main():
    """Main entry point"""
    args = parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)


if __name__ == '__main__':
    main()
