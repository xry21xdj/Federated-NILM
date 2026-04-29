"""
Configuration file for FL-NILM
"""
import torch

class Config:
    # ============ Data Paths ============
    PLAID_DATA_PATH = './data/PLAID'
    PROCESSED_DATA_PATH = './data/processed'
    SPLITS_PATH = './data/splits'
    MODEL_SAVE_PATH = './checkpoints'
    
    # ============ VI Trajectory Parameters ============
    IMAGE_SIZE = 192
    SAMPLES_PER_CYCLE = 128
    SAMPLING_RATE = 30000
    POWER_FREQUENCY = 60
    SIGMOID_MU = 5000
    
    # ============ Model Parameters ============
    NUM_CLASSES = None

    # Available models
    AVAILABLE_MODELS = ['cnn', 'resnet18', 'mobilenet', 'vgg16', 'cnn_small','cnn_medium']
    MODEL_NAME = 'cnn'  # Default model
    
    # Model-specific configs
    MODEL_CONFIGS = {
        'cnn': {
            'conv_filters': [32, 32, 32, 32],
            'fc_units': 128,
            'dropout': 0.5
        },
        'resnet18': {
            'pretrained': False,
        },
        'mobilenet': {
            'width_mult': 1.0,
        },
        'vgg16': {
            'batch_norm': True,
        }
    }
    
    # ============ Training Parameters ============
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    TEACHER_LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # ============ Federated Learning Parameters ============
    NUM_CLIENTS = 100
    NUM_ROUNDS = 100
    KD_START_ROUND = 100
    LOCAL_EPOCHS = 2
    CLIENT_FRACTION = 1.0  # Fraction of clients to use per round
    CLIENT_SELECTION_MODE = 'random'  # random or fixed
    
    # Available aggregation methods
    AVAILABLE_AGGREGATIONS = ['fedavg', 'fedprox', 'scaffold', 'fednova', 'perfednilm', 'olfed', 'nofl','fedkd_prox']

    AGGREGATION_METHOD = 'fedavg'  # Default aggregation
    
    # Aggregation-specific configs
    FEDPROX_MU = 0.01  # FedProx proximal term (also used by PerFedNILM)
    SCAFFOLD_LR = 1.0  # SCAFFOLD learning rate
    
    # PerFedNILM-specific
    USE_PERSONALIZATION = False  # Enable personalized models evaluation
    
    # Non-IID settings
    CLIENT_OVERLAP_RATIO = 0.2
    MIN_SAMPLES_PER_CLIENT = 100

    # OLFED-specific
    USE_KNOWLEDGE_DISTILLATION = False
    KD_ALPHA = 0.7  # KD loss balancing weight
    KD_TEMPERATURE = 3.0  # KD softmax temperature
    TEACHER_MODEL = 'cnn'  # Teacher model architecture
    STUDENT_MODEL = 'cnn_small'  # Student model architecture (lighter)
    
    # ============ Differential Privacy ============
    USE_DP = False
    DP_EPSILON = 100
    DP_DELTA = 1e-2
    DP_CLIP_NORM = 5.0
    
    # ============ Device Configuration ============
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 2
    
    # ============ Evaluation & Logging ============
    EVAL_INTERVAL = 10
    SAVE_INTERVAL = 50
    VERBOSE = True
    
    # WandB configuration
    USE_WANDB = True
    WANDB_PROJECT = 'FL-NILM'
    WANDB_ENTITY = ''
    
    # ============ Random Seed ============
    SEED = 42
    
    @classmethod
    def update_num_classes(cls, num_classes):
        """Update number of classes based on appliances"""
        cls.NUM_CLASSES = num_classes
    
    @classmethod
    def get_experiment_name(cls):
        """Generate experiment name based on configuration"""
        name_parts = [
            f"model_{cls.MODEL_NAME}",
            f"agg_{cls.AGGREGATION_METHOD}",
            f"clients_{cls.NUM_CLIENTS}",
            f"fraction_{cls.CLIENT_FRACTION}",
            f"rounds_{cls.NUM_ROUNDS}",
            f"localE_{cls.LOCAL_EPOCHS}",
            f"lr_{cls.LEARNING_RATE}",
            f"bs_{cls.BATCH_SIZE}",
            f"CMode{cls.CLIENT_SELECTION_MODE}",
        ]
        
        if cls.USE_DP:
            name_parts.append(f"dp_eps_{cls.DP_EPSILON}_delta_{cls.DP_DELTA}_clip_{cls.DP_CLIP_NORM}")
        
        if cls.AGGREGATION_METHOD == 'fedprox':
            name_parts.append(f"mu_{cls.FEDPROX_MU}")
        if cls.AGGREGATION_METHOD == 'fedkd_prox':
            name_parts.append(f"mu_{cls.FEDPROX_MU}_Tlr_{cls.TEACHER_LEARNING_RATE}_KDStart{cls.KD_START_ROUND}")
        if cls.AGGREGATION_METHOD == 'olfed':
            name_parts.append(f"Tlr_{cls.TEACHER_LEARNING_RATE}_KDStart{cls.KD_START_ROUND}")
        if cls.AGGREGATION_METHOD == 'perfednilm':
            name_parts.append(f"mu_{cls.FEDPROX_MU}")
            if cls.USE_PERSONALIZATION:
                name_parts.append("personalized")
        
        return "_".join(name_parts)
    
    @classmethod
    def calculate_dp_noise_scale(cls):
        """
        Calculate DP noise scale (Sigma for Sum) based on epsilon and delta
        """
        import numpy as np

        # Sensitivity is C (for the Sum); do not divide by client count here.
        sensitivity = cls.DP_CLIP_NORM

        # Standard Gaussian noise sigma for the Sum.
        noise_scale = (1 / cls.DP_EPSILON) * sensitivity * np.sqrt(2 * np.log(1.25 / cls.DP_DELTA))
        
        return noise_scale