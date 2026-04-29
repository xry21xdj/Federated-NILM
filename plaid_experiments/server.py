"""
Federated Learning Server with WandB integration
"""
import copy
import os
import re

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from aggregators import create_aggregator, add_differential_privacy_noise


def _sanitize_class_name(name):
    """Make a class name safe for use in a WandB metric key."""
    return re.sub(r'[^a-z0-9]+', '_', str(name).lower()).strip('_') or 'class'


class FLServer:
    """Federated Learning Server"""

    def __init__(self, model, config, wandb_run=None, class_names=None):
        """
        Initialize FL server

        Args:
            model: Global model instance
            config: Configuration object
            wandb_run: WandB run object (optional)
            class_names: Ordered list of appliance names (index matches class label).
                When provided together with a wandb_run, per-class precision /
                recall / F1 are logged during each evaluation.
        """
        self.config = config
        self.device = config.DEVICE
        self.wandb_run = wandb_run
        self.class_names = class_names

        # Global model
        self.global_model = model.to(self.device)

        # Create aggregator
        self.aggregator = create_aggregator(config.AGGREGATION_METHOD, config)

        # DP noise scale
        if config.USE_DP:
            self.dp_noise_scale = config.calculate_dp_noise_scale()
            print(f"\nDifferential Privacy enabled:")
            print(f"  Epsilon: {config.DP_EPSILON}")
            print(f"  Delta: {config.DP_DELTA}")
            print(f"  Noise scale: {self.dp_noise_scale:.6f}")
        else:
            self.dp_noise_scale = 0.0

        # Training history
        self.history = {
            'round': [],
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
        }

        print(f"\nFL Server initialized:")
        print(f"  Model: {config.MODEL_NAME}")
        print(f"  Aggregation: {self.aggregator.name}")
        print(f"  Classes: {config.NUM_CLASSES}")

    def get_global_model_state(self):
        """Get global model state dict"""
        return copy.deepcopy(self.global_model.state_dict())

    def update_global_model(self, state_dict):
        """Update global model with aggregated state"""
        self.global_model.load_state_dict(state_dict)

    def _clip_model_update(self, local_state, global_state):
        """
        L2-norm clipping of the local model update (core DP step).

        Returns:
            clipped_local_state: Local model state after clipping the update.
            total_norm: L2 norm of the unclipped update (for logging).
        """
        clip_norm = self.config.DP_CLIP_NORM

        # 1. Compute the per-parameter update: delta = local - global.
        update_dict = {}
        total_norm_sq = 0.0

        for name, param in local_state.items():
            if name in global_state:
                # Only clip floating-point tensors; skip e.g. BatchNorm running stats
                # which are integer counters or buffers.
                if isinstance(param, torch.Tensor) and param.is_floating_point():
                    delta = param - global_state[name].to(param.device)
                    update_dict[name] = delta
                    total_norm_sq += torch.sum(delta ** 2).item()

        total_norm = np.sqrt(total_norm_sq)

        # 2. Compute the scaling factor.
        # scaling_factor = max(1, norm / C), so update / scaling_factor
        # is equivalent to: scale = min(1, C / norm), then update * scale.
        scaling_factor = min(1.0, clip_norm / (total_norm + 1e-6))

        # 3. Apply clipping and reconstruct: local_new = global + clipped_delta.
        clipped_local_state = copy.deepcopy(local_state)

        for name, delta in update_dict.items():
            clipped_delta = delta * scaling_factor
            clipped_local_state[name] = global_state[name].to(delta.device) + clipped_delta

        return clipped_local_state, total_norm

    def federated_round(self, clients, round_num):
        """
        Execute one federated round (supports FedAvg, FedProx, PerFedNILM, OLFED,
        FedKD-Prox, NoFL with optional Differential Privacy).
        """
        print(f"\n{'='*60}")
        print(f"Round {round_num}/{self.config.NUM_ROUNDS}")
        print(f"{'='*60}")

        # Snapshot of the previous global model (used by PerFedNILM and FedProx).
        prev_global_state = self.get_global_model_state()

        # Client training
        client_states = []
        client_weights = []
        client_ids = []
        client_metrics = []

        wandb_log_dict = {'round': round_num}
        # Per-round DP statistics (raw update norms before clipping).
        dp_norms = []

        print("Client Training:")
        for client in tqdm(clients, desc="Training"):
            # Update client with global model (skipped for NoFL baseline).
            if self.config.AGGREGATION_METHOD not in ['nofl']:
                client.set_model_state(prev_global_state)

            # Local training
            metrics = client.train_local(
                epochs=self.config.LOCAL_EPOCHS,
                global_model_state=prev_global_state if self.config.AGGREGATION_METHOD in ['fedprox', 'perfednilm', 'fedkd_prox'] else None,
                num_round=round_num if self.config.AGGREGATION_METHOD in ['olfed', 'fedkd_prox'] else None,
            )

            local_state = client.get_model_state()
            # DP gradient clipping: clip the per-client update before aggregation.
            if self.config.USE_DP:
                local_state, update_norm = self._clip_model_update(local_state, prev_global_state)
                dp_norms.append(update_norm)

            # Collect results
            client_states.append(local_state)
            client_weights.append(client.num_samples)
            client_ids.append(client.client_id)
            client_metrics.append(metrics)

            # Per-client logging
            wandb_log_dict[f'client_{client.client_id}/train_loss'] = metrics['loss']
            wandb_log_dict[f'client_{client.client_id}/train_acc'] = metrics['accuracy']
            # OLFED-specific: log Teacher metrics.
            if self.config.AGGREGATION_METHOD in ['olfed', 'fedkd_prox']:
                wandb_log_dict[f'client_{client.client_id}/teacher_loss'] = metrics.get('teacher_loss', 0)
                wandb_log_dict[f'client_{client.client_id}/teacher_acc'] = metrics.get('teacher_accuracy', 0)

        # Optional: log DP clipping statistics to WandB.
        if self.config.USE_DP and self.wandb_run:
            wandb_log_dict['dp/avg_update_norm'] = np.mean(dp_norms)
            wandb_log_dict['dp/max_update_norm'] = np.max(dp_norms)
            wandb_log_dict['dp/clip_threshold'] = self.config.DP_CLIP_NORM

        # Aggregate models
        print("Aggregating models...")

        client_states_cpu = [
            {k: v.cpu() for k, v in state.items()}
            for state in client_states
        ]

        # PerFedNILM uses a personalized aggregation that returns extra state.
        if self.config.AGGREGATION_METHOD == 'perfednilm':
            agg_result = self.aggregator.aggregate(
                client_states_cpu,
                client_weights,
                client_ids=client_ids,
                prev_global_state={k: v.cpu() for k, v in prev_global_state.items()}
            )

            aggregated_state = agg_result['global_state']
            personalized_states = agg_result.get('personalized_states', {})
            personalization_weights = agg_result.get('personalization_weights', {})

            # Attach the personalized model back to each client (if any).
            for client in clients:
                if client.client_id in personalized_states:
                    pers_state = {k: v.to(self.config.DEVICE) for k, v in personalized_states[client.client_id].items()}
                    client.personalized_model_state = pers_state

                    # Log per-client personalization weight.
                    lambda_i = personalization_weights.get(client.client_id, 0.0)
                    wandb_log_dict[f'client_{client.client_id}/lambda'] = lambda_i
        else:
            aggregated_state = self.aggregator.aggregate(
                client_states_cpu,
                client_weights
            )

        # Add DP noise if enabled
        if self.config.USE_DP:
            num_clients_k = len(clients)
            aggregated_state = add_differential_privacy_noise(
                aggregated_state,
                self.dp_noise_scale,
                num_clients_k
            )

        # Move back to device and update global model
        aggregated_state = {k: v.to(self.device) for k, v in aggregated_state.items()}
        self.update_global_model(aggregated_state)

        # Calculate metrics
        avg_train_loss = np.mean([m['loss'] for m in client_metrics])
        avg_train_acc = np.mean([m['accuracy'] for m in client_metrics])

        # Record history
        self.history['round'].append(round_num)
        self.history['train_loss'].append(avg_train_loss)
        self.history['train_acc'].append(avg_train_acc)

        # Log to wandb
        if self.wandb_run:
            wandb_log_dict['global/train_loss'] = avg_train_loss
            wandb_log_dict['global/train_accuracy'] = avg_train_acc

            # OLFED: log average Teacher metrics.
            if self.config.AGGREGATION_METHOD in ['olfed', 'fedkd_prox']:
                avg_teacher_loss = np.mean([m.get('teacher_loss', 0) for m in client_metrics])
                avg_teacher_acc = np.mean([m.get('teacher_accuracy', 0) for m in client_metrics])
                wandb_log_dict['global/teacher_train_loss'] = avg_teacher_loss
                wandb_log_dict['global/teacher_train_acc'] = avg_teacher_acc

            self.wandb_run.log(wandb_log_dict, step=round_num)

        print(f"\nRound {round_num} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Train Acc: {avg_train_acc:.2f}%")
        # OLFED-specific extra print.
        if self.config.AGGREGATION_METHOD in ['olfed', 'fedkd_prox']:
            avg_teacher_acc = np.mean([m.get('teacher_accuracy', 0) for m in client_metrics])
            print(f"  Teacher Train Acc: {avg_teacher_acc:.2f}%")

        return {
            'round': round_num,
            'train_loss': avg_train_loss,
            'train_acc': avg_train_acc,
        }

    def evaluate(self, clients, round_num=None):
        """
        Evaluate the global model (and personalized / teacher models when applicable).
        """
        global_state = self.get_global_model_state()

        test_losses = []
        test_accs = []
        personalized_test_losses = []
        personalized_test_accs = []
        # Aggregate predictions across clients for per-class metrics.
        global_y_true_chunks = []
        global_y_pred_chunks = []

        wandb_eval_dict = {}
        if round_num is not None:
            wandb_eval_dict['round'] = round_num

        print(f"Evaluating on {len(clients)} clients...")

        for client in clients:
            # Evaluate the global model on this client's local test set.
            if self.config.AGGREGATION_METHOD not in ['nofl']:
                client.set_model_state(global_state)
            metrics = client.evaluate()

            test_losses.append(metrics['loss'])
            test_accs.append(metrics['accuracy'])

            # Accumulate predictions for per-class metrics (global / student only).
            if 'y_true' in metrics and len(metrics['y_true']) > 0:
                global_y_true_chunks.append(metrics['y_true'])
                global_y_pred_chunks.append(metrics['y_pred'])

                # Per-client per-class support count, for Non-IID composition visibility.
                if self.class_names is not None:
                    counts = np.bincount(metrics['y_true'], minlength=len(self.class_names))
                    for i, name in enumerate(self.class_names):
                        key = _sanitize_class_name(name)
                        wandb_eval_dict[f'client_{client.client_id}/per_class/{key}/support'] = int(counts[i])

            wandb_eval_dict[f'client_{client.client_id}/test_loss'] = metrics['loss']
            wandb_eval_dict[f'client_{client.client_id}/test_acc'] = metrics['accuracy']

            # Personalized model evaluation (PerFedNILM).
            if self.config.AGGREGATION_METHOD == 'perfednilm' and hasattr(client, 'personalized_model_state'):
                client.set_model_state(client.personalized_model_state)
                pers_metrics = client.evaluate()

                personalized_test_losses.append(pers_metrics['loss'])
                personalized_test_accs.append(pers_metrics['accuracy'])

                wandb_eval_dict[f'client_{client.client_id}/pers_test_loss'] = pers_metrics['loss']
                wandb_eval_dict[f'client_{client.client_id}/pers_test_acc'] = pers_metrics['accuracy']
            # OLFED: also evaluate the local Teacher (which acts as the personalized model).
            elif self.config.AGGREGATION_METHOD in ['olfed', 'fedkd_prox']:
                teacher_metrics = client.evaluate_teacher()

                if teacher_metrics:
                    personalized_test_losses.append(teacher_metrics['loss'])
                    personalized_test_accs.append(teacher_metrics['accuracy'])

                    wandb_eval_dict[f'client_{client.client_id}/teacher_test_loss'] = teacher_metrics['loss']
                    wandb_eval_dict[f'client_{client.client_id}/teacher_test_acc'] = teacher_metrics['accuracy']

                    # Personalization gain (Teacher accuracy minus Student accuracy).
                    acc_gain = teacher_metrics['accuracy'] - metrics['accuracy']
                    wandb_eval_dict[f'client_{client.client_id}/personalization_gain'] = acc_gain

        avg_test_loss = np.mean(test_losses)
        avg_test_acc = np.mean(test_accs)

        # Record history
        self.history['test_loss'].append(avg_test_loss)
        self.history['test_acc'].append(avg_test_acc)

        # Per-class precision / recall / F1 on the aggregated predictions (global model).
        if self.class_names is not None and global_y_true_chunks:
            num_classes = len(self.class_names)
            y_true_flat = np.concatenate(global_y_true_chunks)
            y_pred_flat = np.concatenate(global_y_pred_chunks)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true_flat, y_pred_flat,
                labels=list(range(num_classes)),
                zero_division=0,
            )

            for i, name in enumerate(self.class_names):
                key = _sanitize_class_name(name)
                wandb_eval_dict[f'per_class/{key}/precision'] = float(precision[i])
                wandb_eval_dict[f'per_class/{key}/recall'] = float(recall[i])
                wandb_eval_dict[f'per_class/{key}/f1'] = float(f1[i])
                wandb_eval_dict[f'per_class/{key}/support'] = int(support[i])
            wandb_eval_dict['per_class_macro/precision'] = float(precision.mean())
            wandb_eval_dict['per_class_macro/recall'] = float(recall.mean())
            wandb_eval_dict['per_class_macro/f1'] = float(f1.mean())

        result = {
            'test_loss': avg_test_loss,
            'test_acc': avg_test_acc,
            'client_accs': test_accs
        }

        # Personalized aggregate metrics, if applicable.
        if personalized_test_losses:
            avg_pers_test_loss = np.mean(personalized_test_losses)
            avg_pers_test_acc = np.mean(personalized_test_accs)

            result['personalized_test_loss'] = avg_pers_test_loss
            result['personalized_test_acc'] = avg_pers_test_acc
            result['personalized_client_accs'] = personalized_test_accs

            wandb_eval_dict['global/pers_test_loss'] = avg_pers_test_loss
            wandb_eval_dict['global/pers_test_accuracy'] = avg_pers_test_acc

            print(f"Personalized Test Loss: {avg_pers_test_loss:.4f}")
            print(f"Personalized Test Acc: {avg_pers_test_acc:.2f}%")

        # Log to wandb
        if self.wandb_run and round_num is not None:
            wandb_eval_dict['global/test_loss'] = avg_test_loss
            wandb_eval_dict['global/test_accuracy'] = avg_test_acc
            self.wandb_run.log(wandb_eval_dict, step=round_num)

        return result

    def save_checkpoint(self, save_dir, round_num=None):
        """Save server checkpoint"""
        os.makedirs(save_dir, exist_ok=True)

        filename = f"global_model_round_{round_num}.pt" if round_num else "global_model.pt"
        filepath = os.path.join(save_dir, filename)

        checkpoint = {
            'model_state_dict': self.global_model.state_dict(),
            'config': {
                'model_name': self.config.MODEL_NAME,
                'aggregation': self.config.AGGREGATION_METHOD,
                'num_classes': self.config.NUM_CLASSES,
            },
            'history': self.history,
        }

        if round_num:
            checkpoint['round'] = round_num

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")


class FederatedTrainer:
    """Coordinates federated training"""

    def __init__(self, server, client_manager, config, wandb_run=None):
        """
        Initialize federated trainer

        Args:
            server: FLServer instance
            client_manager: ClientManager instance
            config: Configuration object
            wandb_run: WandB run object
        """
        self.server = server
        self.client_manager = client_manager
        self.config = config
        self.wandb_run = wandb_run

    def train(self):
        """Execute federated training"""

        print("\n" + "="*60)
        print("Starting Federated Learning Training")
        print("="*60)
        print(f"Model: {self.config.MODEL_NAME}")
        print(f"Aggregation: {self.config.AGGREGATION_METHOD}")
        print(f"Clients: {self.config.NUM_CLIENTS}")
        print(f"Rounds: {self.config.NUM_ROUNDS}")
        print(f"Local Epochs: {self.config.LOCAL_EPOCHS}")

        # OLFED-specific extra info.
        if self.config.AGGREGATION_METHOD in ['olfed', 'fedkd_prox']:
            print(f"Teacher Model: {self.config.TEACHER_MODEL}")
            print(f"Student Model: {self.config.STUDENT_MODEL}")
            print(f"KD Alpha: {self.config.KD_ALPHA}")
            print(f"KD Temperature: {self.config.KD_TEMPERATURE}")

        print("="*60)

        best_acc = 0.0
        best_pers_acc = 0.0  # Best personalized accuracy seen so far.

        for round_num in range(1, self.config.NUM_ROUNDS + 1):
            # Select clients
            selected_clients = self.client_manager.select_clients(
                self.config.CLIENT_FRACTION, mode=self.config.CLIENT_SELECTION_MODE
            )

            # Training round (per-round WandB logging is handled inside server.federated_round).
            round_metrics = self.server.federated_round(selected_clients, round_num)

            # Evaluation
            if round_num % self.config.EVAL_INTERVAL == 0:
                print("\nEvaluating...")

                # Pass round_num so evaluate() can log per-round metrics itself.
                if self.config.CLIENT_SELECTION_MODE == 'random':
                    eval_metrics = self.server.evaluate(
                        self.client_manager.get_all_clients(),
                        round_num=round_num
                    )
                else:
                    eval_metrics = self.server.evaluate(
                        selected_clients,
                        round_num=round_num
                    )

                print(f"Test Loss: {eval_metrics['test_loss']:.4f}")
                print(f"Test Acc: {eval_metrics['test_acc']:.2f}%")

                # Save best model
                if eval_metrics['test_acc'] > best_acc:
                    best_acc = eval_metrics['test_acc']
                    self.server.save_checkpoint(
                        self.config.MODEL_SAVE_PATH,
                        round_num=None  # Save as best model
                    )
                    if self.wandb_run:
                        self.wandb_run.log({
                            'round': round_num,
                            'best/global_test_acc': best_acc
                        }, step=round_num)

                # Save best personalized model.
                if 'personalized_test_acc' in eval_metrics:
                    pers_acc = eval_metrics['personalized_test_acc']

                    if pers_acc > best_pers_acc:
                        best_pers_acc = pers_acc

                        if self.wandb_run:
                            self.wandb_run.log({
                                'round': round_num,
                                'best/personalized_test_acc': best_pers_acc
                            }, step=round_num)

            # Save checkpoint (periodic)
            if round_num % self.config.SAVE_INTERVAL == 0:
                self.server.save_checkpoint(
                    self.config.MODEL_SAVE_PATH,
                    round_num=round_num
                )

        print("\n" + "="*60)
        print("Training Completed!")
        print(f"Best Test Accuracy: {best_acc:.2f}%")
        print("="*60)

        return self.server.history
