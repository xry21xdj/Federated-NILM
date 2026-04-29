"""
Federated Learning Client for NILM
"""
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class NILMDataset(Dataset):
    """Dataset for NILM VI trajectory images"""

    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample['image']
        label = sample['label']

        # Convert to tensor
        if not isinstance(image, torch.Tensor):
            image = torch.FloatTensor(image)

        # Ensure correct shape [C, H, W]
        if image.ndim == 2:
            image = image.unsqueeze(0)

        return image, label


class FLClient:
    """Federated Learning Client"""

    def __init__(self, client_id, model, train_data, test_data, config):
        """
        Initialize FL client

        Args:
            client_id: Client identifier
            model: Model instance
            train_data: Training data list
            test_data: Test data list
            config: Configuration object
        """
        self.client_id = client_id
        self.config = config
        self.device = config.DEVICE

        # OLFED / FedKD-Prox use a teacher-student two-model setup; other methods use a single model.
        if config.AGGREGATION_METHOD in ['olfed', 'fedkd_prox']:
            from models import create_model

            # Teacher model: larger, kept locally for personalization.
            self.teacher_model = create_model(
                config.TEACHER_MODEL,
                config.NUM_CLASSES,
                config.MODEL_CONFIGS.get(config.TEACHER_MODEL, {})
            ).to(self.device)

            # Student model: smaller, used for federated communication.
            self.model = model.to(self.device)

            # Teacher optimizer (typically uses a smaller learning rate than the student).
            self.teacher_optimizer = optim.Adam(
                self.teacher_model.parameters(),
                lr=config.TEACHER_LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
        else:
            self.model = model.to(self.device)

        # Data loaders
        self.train_loader = DataLoader(
            NILMDataset(train_data),
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False
        )

        self.test_loader = DataLoader(
            NILMDataset(test_data),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Training history
        self.num_samples = len(train_data)

    def set_model_state(self, state_dict):
        """Load global model state"""
        self.model.load_state_dict(state_dict)

    def get_model_state(self):
        """Get current model state"""
        return copy.deepcopy(self.model.state_dict())

    def train_local(self, epochs, global_model_state, num_round):
        """
        Local training. OLFED / FedKD-Prox use a dedicated teacher-student loop;
        other methods use the standard local update.

        Args:
            epochs: Number of local epochs
            global_model_state: Global model state (for FedProx / PerFedNILM / FedKD-Prox)
            num_round: Current global round (used by OLFED / FedKD-Prox to gate KD)

        Returns:
            Training metrics
        """
        if self.config.AGGREGATION_METHOD in ['olfed', 'fedkd_prox']:
            return self._train_olfed(epochs, global_model_state, num_round)
        else:
            return self._train_standard(epochs, global_model_state)

    def _train_standard(self, epochs, global_model_state=None):
        """
        Standard local training (used by FedAvg, FedProx, SCAFFOLD, FedNova,
        PerFedNILM, NoFL).

        Args:
            epochs: Number of local epochs
            global_model_state: Global model state (only used for FedProx-style proximal term)

        Returns:
            Training metrics
        """
        self.model.train()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        for epoch in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                # Add FedProx proximal term if using FedProx / PerFedNILM.
                if (self.config.AGGREGATION_METHOD in ['fedprox', 'perfednilm'] and
                        global_model_state is not None):
                    proximal_term = 0.0
                    for (name, param), (_, global_param) in zip(
                        self.model.named_parameters(),
                        global_model_state.items()
                    ):
                        if 'weight' in name or 'bias' in name:
                            proximal_term += torch.norm(param - global_param.to(self.device)) ** 2

                    loss += (self.config.FEDPROX_MU / 2) * proximal_term

                loss.backward()
                self.optimizer.step()

                # Statistics
                total_loss += loss.item() * data.size(0)
                total_correct += (output.argmax(1) == target).sum().item()
                total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        avg_accuracy = 100.0 * total_correct / total_samples

        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'samples': total_samples
        }

    def _train_olfed(self, epochs, global_model_state, num_round):
        """
        OLFED training procedure:
          1. Train the local Teacher model on real labels.
          2. Train the Student model with knowledge distillation from the Teacher
             (only after KD_START_ROUND; before that, the Student also uses CE).
        """

        # ---- Step 1: train the Teacher model ----
        self.teacher_model.train()
        teacher_loss_sum = 0.0
        teacher_correct = 0
        teacher_total = 0

        for epoch in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Teacher forward / backward
                self.teacher_optimizer.zero_grad()
                outputs = self.teacher_model(data)
                loss = self.criterion(outputs, target)

                loss.backward()
                self.teacher_optimizer.step()

                teacher_loss_sum += loss.item() * data.size(0)
                teacher_correct += (outputs.argmax(1) == target).sum().item()
                teacher_total += data.size(0)

        teacher_loss = teacher_loss_sum / teacher_total
        teacher_acc = 100.0 * teacher_correct / teacher_total

        # ---- Step 2: train the Student via knowledge distillation ----
        self.model.train()
        self.teacher_model.eval()  # Freeze the teacher during student training.

        student_loss_sum = 0.0
        student_correct = 0
        student_total = 0

        for epoch in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                # Student forward pass
                student_outputs = self.model(data)

                if num_round > self.config.KD_START_ROUND:
                    # Teacher forward pass (no gradients).
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(data)

                    # Knowledge distillation loss
                    loss = self._kd_loss(
                        student_outputs,
                        teacher_outputs,
                        target,
                        self.config.KD_ALPHA,
                        self.config.KD_TEMPERATURE
                    )
                else:
                    loss = self.criterion(student_outputs, target)

                # Add FedProx proximal term for FedKD-Prox.
                if (self.config.AGGREGATION_METHOD in ['fedkd_prox'] and
                        global_model_state is not None):
                    proximal_term = 0.0
                    for (name, param), (_, global_param) in zip(
                        self.model.named_parameters(),
                        global_model_state.items()
                    ):
                        if 'weight' in name or 'bias' in name:
                            proximal_term += torch.norm(param - global_param.to(self.device)) ** 2

                    loss += (self.config.FEDPROX_MU / 2) * proximal_term

                loss.backward()
                self.optimizer.step()

                student_loss_sum += loss.item() * data.size(0)
                student_correct += (student_outputs.argmax(1) == target).sum().item()
                student_total += data.size(0)

        student_loss = student_loss_sum / student_total
        student_acc = 100.0 * student_correct / student_total

        return {
            'loss': student_loss,
            'accuracy': student_acc,
            'samples': student_total,
            'teacher_loss': teacher_loss,
            'teacher_accuracy': teacher_acc
        }

    def _kd_loss(self, student_logits, teacher_logits, labels, alpha, temperature):
        """
        Knowledge distillation loss (Eq. 2 in the OLFED paper):

            L_KD = (1 - alpha) * L_CE(Student, y) + alpha * L_KL(Student/T, Teacher/T)
        """
        # Hard-label loss: Student against ground-truth labels.
        ce_loss = self.criterion(student_logits, labels)

        # Soft-label loss: Student matches Teacher's distribution.
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        kl_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (temperature ** 2)

        # Weighted combination.
        total_loss = (1 - alpha) * ce_loss + alpha * kl_loss

        return total_loss

    def evaluate(self):
        """
        Evaluate the (student / global) model on this client's test set.

        Returns a dict with aggregate loss/accuracy and the per-sample true
        labels and predicted labels (as numpy arrays), so that downstream code
        can aggregate them across clients for per-class metrics.
        """
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_true = []
        all_pred = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                preds = output.argmax(1)
                correct = (preds == target).sum().item()
                all_true.append(target.cpu().numpy())
                all_pred.append(preds.cpu().numpy())

                total_loss += loss.item() * data.size(0)
                total_correct += correct
                total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        avg_accuracy = 100.0 * total_correct / total_samples

        y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
        y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)

        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def evaluate_teacher(self):
        """
        Evaluate the Teacher model (used as the personalized model in OLFED / FedKD-Prox).
        Returns the same keys as `evaluate()` for symmetry, or None if no teacher exists.
        """
        if not hasattr(self, 'teacher_model'):
            return None

        self.teacher_model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_true = []
        all_pred = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.teacher_model(data)
                loss = self.criterion(output, target)
                preds = output.argmax(1)
                correct = (preds == target).sum().item()
                all_true.append(target.cpu().numpy())
                all_pred.append(preds.cpu().numpy())

                total_loss += loss.item() * data.size(0)
                total_correct += correct
                total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        avg_accuracy = 100.0 * total_correct / total_samples

        y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
        y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)

        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'y_true': y_true,
            'y_pred': y_pred,
        }


class ClientManager:
    """Manage multiple FL clients"""

    def __init__(self, client_splits, model_fn, config):
        """
        Initialize client manager

        Args:
            client_splits: List of client data splits
            model_fn: Function that returns a model instance
            config: Configuration object
        """
        self.config = config
        self.clients = []
        self.rng = random.Random(config.SEED)

        print(f"\nInitializing {len(client_splits)} FL Clients...")
        print("=" * 60)

        for split in client_splits:
            model = model_fn()
            client = FLClient(
                client_id=split['client_id'],
                model=model,
                train_data=split['train'],
                test_data=split['test'],
                config=config
            )
            self.clients.append(client)

            print(f"Client {split['client_id']}: "
                  f"{len(split['train'])} train, {len(split['test'])} test samples")

        print("=" * 60)

    def get_client(self, client_id):
        """Get client by ID"""
        for client in self.clients:
            if client.client_id == client_id:
                return client
        return None

    def get_all_clients(self):
        """Get all clients"""
        return self.clients

    def select_clients(self, fraction=1.0, mode='random'):
        """
        Select clients for the current round.

        Args:
            fraction: Participation fraction (e.g., 0.1 for 10%).
            mode:
                'random': sample a fresh subset each round (typical FedAvg).
                'fixed':  always use the full client set (stable network simulation).
        """
        num_selected = max(1, int(len(self.clients) * fraction))

        if mode == 'random':
            return self.rng.sample(self.clients, num_selected)

        elif mode == 'fixed':
            return self.clients

        else:
            raise ValueError("Mode must be 'random' or 'fixed'")
