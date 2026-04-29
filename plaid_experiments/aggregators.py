"""
Federated Learning Aggregation Algorithms
Supports: FedAvg, FedProx, SCAFFOLD, FedNova, PerFedNILM
"""
import torch
import copy
import numpy as np


class BaseAggregator:
    """Base class for federated aggregation"""
    
    def __init__(self, config):
        self.config = config
    
    def aggregate(self, client_states, client_weights=None):
        """
        Aggregate client models
        
        Args:
            client_states: List of client model state dicts
            client_weights: Optional weights for weighted averaging
        
        Returns:
            Aggregated global model state dict
        """
        raise NotImplementedError


class FedAvgAggregator(BaseAggregator):
    """
    Federated Averaging (FedAvg)
    McMahan et al., 2017
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "FedAvg"
    
    def aggregate(self, client_states, client_weights=None):
        """Standard FedAvg weighted averaging"""
        
        if client_weights is None:
            # Equal weights if not specified
            client_weights = [1.0 / len(client_states)] * len(client_states)
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Initialize global state
        global_state = {}
        
        # Get keys from first client
        keys = client_states[0].keys()
        
        # Weighted average of parameters
        for key in keys:
            global_state[key] = sum(
                client_weights[i] * client_states[i][key].float()
                for i in range(len(client_states))
            )
        
        return global_state


class FedProxAggregator(BaseAggregator):
    """
    Federated Proximal (FedProx)
    Li et al., 2020
    
    Note: The proximal term is applied during local training,
    aggregation is same as FedAvg
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "FedProx"
        self.mu = config.FEDPROX_MU
    
    def aggregate(self, client_states, client_weights=None):
        """Same as FedAvg for aggregation"""
        fedavg = FedAvgAggregator(self.config)
        return fedavg.aggregate(client_states, client_weights)
    
    def get_proximal_term(self, local_model, global_model):
        """
        Calculate FedProx proximal term
        
        Args:
            local_model: Local model state dict
            global_model: Global model state dict
        
        Returns:
            Proximal term value
        """
        proximal_term = 0.0
        
        for key in local_model.keys():
            if 'weight' in key or 'bias' in key:
                proximal_term += torch.norm(
                    local_model[key] - global_model[key]
                ) ** 2
        
        return (self.mu / 2) * proximal_term


class SCAFFOLDAggregator(BaseAggregator):
    """
    SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
    Karimireddy et al., 2020
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "SCAFFOLD"
        self.lr = config.SCAFFOLD_LR
        
        # Control variates (initialized when first used)
        self.server_control = None
        self.client_controls = {}
    
    def aggregate(self, client_states, client_weights=None, 
                  client_controls=None, client_deltas=None):
        """
        SCAFFOLD aggregation with control variates
        
        Args:
            client_states: List of client model states
            client_weights: Client sample weights
            client_controls: List of client control variates
            client_deltas: List of model update deltas
        """
        
        if client_weights is None:
            client_weights = [1.0 / len(client_states)] * len(client_states)
        
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Aggregate model updates
        global_state = {}
        keys = client_states[0].keys()
        
        for key in keys:
            global_state[key] = sum(
                client_weights[i] * client_states[i][key].float()
                for i in range(len(client_states))
            )
        
        # Update server control variate if client controls provided
        if client_controls is not None:
            self._update_server_control(client_controls, client_weights)
        
        return global_state
    
    def _update_server_control(self, client_controls, client_weights):
        """Update server control variate"""
        
        if self.server_control is None:
            # Initialize server control
            self.server_control = {}
            keys = client_controls[0].keys()
            for key in keys:
                self.server_control[key] = torch.zeros_like(client_controls[0][key])
        
        # Update: c = c + 1/N * Σ(c_i^+ - c_i)
        keys = self.server_control.keys()
        for key in keys:
            delta_control = sum(
                client_weights[i] * client_controls[i][key]
                for i in range(len(client_controls))
            )
            self.server_control[key] += delta_control


class FedNovaAggregator(BaseAggregator):
    """
    FedNova: Normalized Averaging for Federated Learning
    Wang et al., 2020
    
    Handles heterogeneous local updates
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "FedNova"
    
    def aggregate(self, client_states, client_weights=None, 
                  client_steps=None, tau_eff=None):
        """
        FedNova aggregation with normalized averaging
        
        Args:
            client_states: List of client model states
            client_weights: Client sample weights
            client_steps: Number of local steps per client
            tau_eff: Effective number of local steps
        """
        
        if client_weights is None:
            client_weights = [1.0 / len(client_states)] * len(client_states)
        
        if client_steps is None:
            # Assume uniform local steps
            client_steps = [1.0] * len(client_states)
        
        if tau_eff is None:
            # Calculate effective tau
            tau_eff = sum(w * s for w, s in zip(client_weights, client_steps))
        
        # Normalize weights by local steps
        total_weight = sum(client_weights)
        normalized_weights = [
            (w / total_weight) * (tau_eff / s)
            for w, s in zip(client_weights, client_steps)
        ]
        
        # Aggregate with normalized weights
        global_state = {}
        keys = client_states[0].keys()
        
        for key in keys:
            global_state[key] = sum(
                normalized_weights[i] * client_states[i][key].float()
                for i in range(len(client_states))
            )
        
        return global_state


class PerFedNILMAggregator(BaseAggregator):
    """
    PerFedNILM: Personalized Federated Learning for NILM
    Pan et al., 2024
    
    Features:
    1. Proximal term to limit local update bias
    2. Personalized models via dynamic weight mixing  
    3. Client dropout handling via historical reweighting
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "PerFedNILM"
        self.mu = config.FEDPROX_MU
        
        # Track client participation
        self.client_participation = {}
        self.total_rounds = 0
        self.client_history = {}
        self.personalization_weights = {}
        
    def aggregate(self, client_states, client_weights=None, 
                  client_ids=None, prev_global_state=None):
        """PerFedNILM aggregation with personalization"""
        
        self.total_rounds += 1
        
        if client_weights is None:
            client_weights = [1.0 / len(client_states)] * len(client_states)
        
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Update participation tracking
        if client_ids is not None:
            for cid in client_ids:
                self.client_participation[cid] = self.client_participation.get(cid, 0) + 1
        
        # Standard aggregation (Eq. 10 in paper)
        global_state = {}
        keys = client_states[0].keys()
        
        for key in keys:
            global_state[key] = sum(
                client_weights[i] * client_states[i][key].float()
                for i in range(len(client_states))
            )
        
        # Handle client dropout (Eq. 11 in paper)
        if client_ids is not None:
            all_known = set(self.client_history.keys())
            current_online = set(client_ids)
            absent = all_known - current_online
            
            if len(absent) > 0 and len(current_online) > 0:
                for key in keys:
                    absent_term = torch.zeros_like(global_state[key])
                    total_absent_weight = 0.0
                    
                    for absent_id in absent:
                        if absent_id in self.client_history:
                            mu_j = self.client_participation.get(absent_id, 0) / self.total_rounds
                            absent_term += mu_j * self.client_history[absent_id][key].float()
                            total_absent_weight += mu_j
                    
                    if total_absent_weight > 0:
                        global_state[key] += absent_term / len(current_online)
        
        # Calculate personalized models (Eq. 9 in paper)
        personalized_states = {}
        
        if client_ids is not None and prev_global_state is not None:
            for i, client_id in enumerate(client_ids):
                lambda_i = self._calculate_lambda(
                    client_states[i], 
                    prev_global_state, 
                    global_state
                )
                
                self.personalization_weights[client_id] = lambda_i
                
                # Build personalized model: ω_i^{t+1} = ω^{t+1} + d_i^t
                personalized_states[client_id] = {}
                for key in keys:
                    g_i = prev_global_state[key] - client_states[i][key]
                    g = prev_global_state[key] - global_state[key]
                    d_i = (-g_i - g) * lambda_i + g
                    personalized_states[client_id][key] = global_state[key] + d_i
        
        # Update history
        if client_ids is not None:
            for i, cid in enumerate(client_ids):
                self.client_history[cid] = copy.deepcopy(client_states[i])
        
        return {
            'global_state': global_state,
            'personalized_states': personalized_states,
            'personalization_weights': self.personalization_weights
        }
    
    def _calculate_lambda(self, local_state, prev_global, new_global):
        """Calculate personalization weight λ_i"""
        local_norm_sq = 0.0
        global_norm_sq = 0.0
        local_global_dot = 0.0
        
        for key in local_state.keys():
            g_i = (prev_global[key] - local_state[key]).float()
            g = (prev_global[key] - new_global[key]).float()
            
            local_norm_sq += torch.sum(g_i ** 2).item()
            global_norm_sq += torch.sum(g ** 2).item()
            local_global_dot += torch.sum(g_i * g).item()
        
        numerator = local_norm_sq + local_global_dot
        denominator = local_norm_sq + 2 * local_global_dot + global_norm_sq
        
        if abs(denominator) > 1e-6:
            lambda_i = numerator / denominator
        else:
            lambda_i = 0.5
        
        return max(0.0, min(1.0, lambda_i))
    
    def get_proximal_term(self, local_model, global_model):
        """FedProx proximal term for local training"""
        proximal_term = 0.0
        
        for key in local_model.keys():
            if 'weight' in key or 'bias' in key:
                proximal_term += torch.norm(
                    local_model[key] - global_model[key]
                ) ** 2
        
        return (self.mu / 2) * proximal_term

class OLFEDAggregator(BaseAggregator):
    """
    OLFED: OnLine FEderated Distillation
    Li et al., 2024
    
    Note: OLFED uses standard FedAvg for aggregating Student models.
    The personalization happens via Teacher models kept locally at clients.
    Knowledge distillation is applied during local training (not aggregation).
    
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "OLFED"
    
    def aggregate(self, client_states, client_weights=None):
        """
        OLFED aggregation (same as FedAvg for Student models)
        
        Only Student models are aggregated and communicated.
        Teacher models remain local for personalization.
        
        Args:
            client_states: List of Student model state dicts
            client_weights: Client sample weights
        
        Returns:
            Aggregated Student model state dict
        """
        fedavg = FedAvgAggregator(self.config)
        return fedavg.aggregate(client_states, client_weights)
    

def create_aggregator(aggregation_method, config):
    """
    Factory function to create aggregators
    
    Args:
        aggregation_method: Name of aggregation method
        config: Configuration object
    
    Returns:
        Aggregator instance
    """
    
    if aggregation_method == 'fedavg':
        return FedAvgAggregator(config)
    if aggregation_method == 'nofl':
        return FedAvgAggregator(config)
    elif aggregation_method == 'fedprox':
        return FedProxAggregator(config)
    
    elif aggregation_method == 'scaffold':
        return SCAFFOLDAggregator(config)
    
    elif aggregation_method == 'fednova':
        return FedNovaAggregator(config)
    
    elif aggregation_method == 'perfednilm':
        return PerFedNILMAggregator(config)
    elif aggregation_method == 'olfed':
        return OLFEDAggregator(config)
    elif aggregation_method == 'fedkd_prox':
        return FedAvgAggregator(config)
    
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")


def add_differential_privacy_noise(model_state, noise_scale, num_clients):
    """
    Add Gaussian noise for differential privacy
    
    Args:
        model_state: Model state dictionary
        noise_scale: Scale of Gaussian noise
    
    Returns:
        Noisy model state
    """
    noisy_state = {}
    
    for key, param in model_state.items():
        noise = torch.randn_like(param) * noise_scale/num_clients
        noisy_state[key] = param + noise
    
    return noisy_state