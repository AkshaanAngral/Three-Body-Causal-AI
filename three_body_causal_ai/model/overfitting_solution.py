# three_body_causal_ai/model/overfitting_solution.py

# Enhanced Overfitting Solution for Three-Body Neural ODE VAE
# This addresses the core issues causing overfitting in your model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, default_collate
import numpy as np
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import pickle
import seaborn as sns
from collections import defaultdict

# --- Path Setup (CRITICAL: These must be at the top level of the script) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT should point to 'three_body_causal_ai/'
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..') 
DATA_ROOT_DIR = os.path.join(PROJECT_ROOT, 'data') # This is the correct absolute path to your data folder
MODELS_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'model', 'checkpoints')
SCALER_PATH = os.path.join(MODELS_CHECKPOINT_DIR, 'robust_scaler.pkl')
# --- END Path Setup ---

# Ensure seaborn style is set up if desired
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# --- Custom Collate Function for DataLoader (CRITICAL FIX) ---
def custom_collate_fn(batch):
    """Custom collate function for DataLoader."""
    sequences = [item[0] for item in batch]
    targets = [item[1] for item in batch] # <--- CRITICAL FIX: Correctly get target (item[1])
    metadata_list = [item[2] for item in batch]
    
    collated_sequences = default_collate(sequences)
    collated_targets = default_collate(targets)
    
    return collated_sequences, collated_targets, metadata_list

# --- Data Analysis Utilities (unchanged, but included for completeness) ---
class DataAnalyzer:
    """Analyze your data to understand potential overfitting causes."""
    
    @staticmethod
    def analyze_data_distribution(data_dir=DATA_ROOT_DIR): # Use DATA_ROOT_DIR
        print("=== DATA DISTRIBUTION ANALYSIS ===")
        clean_files = []
        perturbed_files = []
        clean_dir = os.path.join(data_dir, 'clean')
        perturbed_dir = os.path.join(data_dir, 'perturbed')
        if os.path.exists(clean_dir):
            clean_files = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.npz')]
        if os.path.exists(perturbed_dir):
            perturbed_files = [os.path.join(perturbed_dir, f) for f in os.listdir(perturbed_dir) if f.endswith('.npz')]
        
        print(f"Clean simulations: {len(clean_files)}")
        print(f"Perturbed simulations: {len(perturbed_files)}")
        total_files = len(clean_files) + len(perturbed_files)
        if total_files > 0:
            clean_ratio = len(clean_files) / total_files
            print(f"Clean/Total ratio: {clean_ratio:.2f}")
            if clean_ratio < 0.3 or clean_ratio > 0.7:
                print("⚠️ ISSUE: Unbalanced dataset detected!")
                print("    This can cause overfitting to the majority class.")
                return "unbalanced"
        all_data_stats = []
        for file_path in (clean_files + perturbed_files)[:min(10, total_files)]:
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    solution = data['solution']
                    stats = {
                        'mean': np.mean(solution), 'std': np.std(solution),
                        'min': np.min(solution), 'max': np.max(solution),
                        'length': len(solution)
                    }
                    all_data_stats.append(stats)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        if all_data_stats:
            mean_stats = {key: np.mean([s[key] for s in all_data_stats]) for key in all_data_stats[0].keys()}
            print(f"Average data statistics (from sampled files):")
            for key, value in mean_stats.items():
                print(f"  {key}: {value:.4f}")
        return "balanced"
    
    @staticmethod
    def check_data_quality(data_dir=DATA_ROOT_DIR): # Use DATA_ROOT_DIR
        print("\n=== DATA QUALITY CHECK ===")
        issues = []
        sample_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.npz'):
                    sample_files.append(os.path.join(root, file))
                    if len(sample_files) >= 20: break
            if len(sample_files) >= 20: break
        nan_count = 0; inf_count = 0; zero_variance_count = 0
        for file_path in sample_files:
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    solution = data['solution']
                    if np.any(np.isnan(solution)): nan_count += 1
                    if np.any(np.isinf(solution)): inf_count += 1
                    if np.var(solution) < 1e-10: zero_variance_count += 1
            except Exception as e:
                print(f"Error checking {file_path}: {e}")
        print(f"Files with NaN values (sampled): {nan_count}")
        print(f"Files with infinite values (sampled): {inf_count}")
        print(f"Files with zero variance (sampled): {zero_variance_count}")
        if nan_count > 0: issues.append("nan_values")
        if inf_count > 0: issues.append("infinite_values")
        if zero_variance_count > len(sample_files) * 0.1: issues.append("low_variance")
        return issues


class ImprovedDataset(Dataset):
    """Enhanced dataset with stronger regularization and better preprocessing."""
    
    def __init__(self, data_files, sequence_length=12, prediction_horizon=3, 
                 scaler=None, fit_scaler=False, augment_data=True, 
                 balance_classes=True, validation_split=0.0): # validation_split not used here directly
        
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.augment_data = augment_data
        self.balance_classes = balance_classes
        
        self.sequences = []
        self.targets = []
        self.metadata = []
        self.scaler = scaler
        
        clean_samples, perturbed_samples = self._load_and_segment_data(data_files)
        
        if self.balance_classes and len(clean_samples) > 0 and len(perturbed_samples) > 0:
            min_samples = min(len(clean_samples), len(perturbed_samples))
            min_samples = min(min_samples, 50) # Cap at 50 per class for strong balancing
            clean_samples = clean_samples[:min_samples]
            perturbed_samples = perturbed_samples[:min_samples]
            print(f"Strongly balanced dataset: {min_samples} samples per class (capped for generalization)")
        
        all_samples = clean_samples + perturbed_samples
        
        if fit_scaler and self.scaler is None:
            self._fit_robust_scaler(all_samples)
        
        for sequence, target, metadata in all_samples:
            if self.scaler is not None:
                sequence_norm, target_norm = self._normalize_data(sequence, target)
            else:
                sequence_norm, target_norm = sequence, target
            
            self.sequences.append(sequence_norm)
            self.targets.append(target_norm)
            self.metadata.append(metadata)
            
            if self.augment_data:
                for _ in range(2): # Create 2 augmented versions per sample
                    aug_seq, aug_target = self._strong_augment_sequence(sequence_norm, target_norm)
                    self.sequences.append(aug_seq)
                    self.targets.append(aug_target)
                    
                    aug_metadata = metadata.copy()
                    aug_metadata['augmented'] = True
                    self.metadata.append(aug_metadata)
    
    def _load_and_segment_data(self, data_files):
        clean_samples = []
        perturbed_samples = []
        
        for file_path in data_files:
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    solution = data['solution']
                    
                    # Stronger quality checks
                    if (np.any(np.isnan(solution)) or 
                        np.any(np.isinf(solution)) or 
                        np.var(solution) < 1e-8 or # Stricter variance check
                        np.max(np.abs(solution)) > 100): # Remove extreme outliers
                        continue
                    
                    velocity_variance = np.var(np.diff(solution, axis=0))
                    if velocity_variance < 1e-10: # Check for reasonable dynamics (not too static)
                        continue
                        
                    perturbation_type = str(data.get('perturbation_type', None))
                    is_perturbed = 1 if perturbation_type != 'None' else 0
                    
                    max_start = len(solution) - self.sequence_length - self.prediction_horizon + 1
                    if max_start <= 0:
                        continue
                    
                    stride = max(5, max_start // 10) # At most 10 sequences per file, adjust stride
                    for i in range(0, max_start, stride):
                        sequence = solution[i:i + self.sequence_length]
                        target = solution[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
                        
                        metadata = {
                            'is_perturbed': is_perturbed,
                            'perturbation_type': perturbation_type,
                            'file': file_path,
                            'start_idx': i,
                            'augmented': False
                        }
                        
                        if is_perturbed:
                            perturbed_samples.append((sequence, target, metadata))
                        else:
                            clean_samples.append((sequence, target, metadata))
                            
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        return clean_samples, perturbed_samples
    
    def _fit_robust_scaler(self, all_samples):
        all_data = []
        for seq, target, _ in all_samples:
            all_data.append(seq)
            all_data.append(target)
        
        combined_data = np.concatenate(all_data, axis=0)
        
        # Remove extreme outliers before fitting scaler
        percentile_99 = np.percentile(np.abs(combined_data), 99, axis=0)
        mask = np.all(np.abs(combined_data) <= percentile_99, axis=1)
        filtered_data = combined_data[mask]
        
        self.scaler = RobustScaler()
        self.scaler.fit(filtered_data)
        
        print(f"Scaler fitted on {len(filtered_data)} points (outliers removed)")
        
        # Save scaler
        os.makedirs(MODELS_CHECKPOINT_DIR, exist_ok=True) # Use global path
        with open(SCALER_PATH, 'wb') as f: # Use global path
            pickle.dump(self.scaler, f)
        print(f"✅ Scaler saved to: {SCALER_PATH}")
    
    def _normalize_data(self, sequence, target):
        sequence_norm = self.scaler.transform(sequence)
        target_norm = self.scaler.transform(target)
        
        # Clip extreme values after normalization
        sequence_norm = np.clip(sequence_norm, -5, 5)
        target_norm = np.clip(target_norm, -5, 5)
        
        return sequence_norm, target_norm
    
    def _strong_augment_sequence(self, sequence, target):
        aug_sequence = sequence.copy()
        aug_target = target.copy()
        
        # 1. Gaussian noise
        noise_level = 0.02
        aug_sequence += np.random.normal(0, noise_level, sequence.shape)
        aug_target += np.random.normal(0, noise_level, target.shape)
        
        # 2. Small time shift (if sequence is long enough)
        if len(sequence) > 5:
            shift = np.random.randint(-2, 3)
            if shift != 0:
                aug_sequence = np.roll(aug_sequence, shift, axis=0)
        
        # 3. Small scaling
        scale_factor = np.random.uniform(0.95, 1.05)
        aug_sequence *= scale_factor
        aug_target *= scale_factor
        
        return aug_sequence, aug_target
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor(self.targets[idx])
        metadata = self.metadata[idx]
        return sequence, target, metadata


class UltraMinimalNeuralODE(nn.Module):
    def __init__(self, state_dim=18, latent_dim=3, hidden_dim=16, dropout_rate=0.5):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self.ode_func = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        self.perturbation_classifier = nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def ode_forward(self, t, z):
        return self.ode_func(z)
    
    def forward(self, x_sequence, t_span):
        batch_size, seq_len, state_dim = x_sequence.shape
        
        x_flat = x_sequence.reshape(-1, state_dim)
        z_flat = self.encode(x_flat)
        z_seq = z_flat.reshape(batch_size, seq_len, self.latent_dim)
        
        z0 = z_seq[:, -1, :]
        
        if self.training:
            z_pred_list = []
            z_current = z0
            for _ in range(len(t_span)):
                z_current = z_current + 0.1 * self.ode_func(z_current) # Simple Euler step
                z_pred_list.append(z_current)
            z_pred = torch.stack(z_pred_list, dim=1)
        else:
            try:
                z_pred = odeint(self.ode_forward, z0, t_span, method='euler')
                z_pred = z_pred.permute(1, 0, 2)
            except:
                z_pred = z0.unsqueeze(1).repeat(1, len(t_span), 1)
        
        x_recon = self.decode(z_flat).reshape(batch_size, seq_len, state_dim)
        x_pred = self.decode(z_pred.reshape(-1, self.latent_dim)).reshape(batch_size, len(t_span), state_dim)
        
        z_mean = torch.mean(z_seq, dim=1)
        perturbation_prob = self.perturbation_classifier(z_mean)
        
        return x_recon, x_pred, perturbation_prob, z_seq


class AntiOverfittingTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-2, # Strong weight decay
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5,  min_lr=1e-7
        )
        
        self.patience = 15
        self.best_val_loss = float('inf')
        self.best_ratio = float('inf')
        self.patience_counter = 0
        
        self.train_losses = []
        self.val_losses = []
        self.val_ratios = []
        self.learning_rates = []
    
    def compute_regularized_loss(self, x_recon, x_pred, x_sequence, x_target, 
                                 perturbation_prob, is_perturbed, z_seq, epoch):
        # Basic losses
        recon_loss = nn.MSELoss()(x_recon, x_sequence)
        pred_loss = nn.MSELoss()(x_pred, x_target)
        
        # Classification loss
        perturbation_target = is_perturbed.float().unsqueeze(1)
        class_loss = nn.BCELoss()(perturbation_prob, perturbation_target)
        
        # Regularization losses
        
        # 1. L2 regularization on latent representations (prevent overfitting)
        latent_reg = torch.mean(torch.norm(z_seq, dim=-1))
        
        # 2. Smoothness regularization (encourage smooth latent trajectories)
        if z_seq.size(1) > 1:
            z_diff = z_seq[:, 1:] - z_seq[:, :-1]
            smoothness_reg = torch.mean(torch.norm(z_diff, dim=-1))
        else:
            smoothness_reg = torch.tensor(0.0, device=self.device)
        
        # 3. Prediction consistency (predictions shouldn't be too different from inputs)
        if x_pred.size(1) > 0 and x_sequence.size(1) > 0:
            last_input = x_sequence[:, -1:, :]
            first_pred = x_pred[:, :1, :]
            consistency_loss = nn.MSELoss()(first_pred, last_input)
        else:
            consistency_loss = torch.tensor(0.0, device=self.device)
        
        # Adaptive weighting (stronger regularization early in training)
        reg_weight = max(0.1, 1.0 - epoch / 50.0) # Decreases from 1.0 to 0.1
        
        total_loss = (
            1.0 * recon_loss +
            0.5 * pred_loss +
            1.5 * class_loss + # <--- CRITICAL FIX: Increased weight from 0.3 to 1.0
            reg_weight * 0.1 * latent_reg +
            reg_weight * 0.05 * smoothness_reg +
            reg_weight * 0.2 * consistency_loss
        )
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'prediction': pred_loss,
            'classification': class_loss,
            'latent_reg': latent_reg,
            'smoothness': smoothness_reg,
            'consistency': consistency_loss
        }
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (x_sequence, x_target, metadata) in enumerate(train_loader):
            x_sequence = x_sequence.to(self.device)
            x_target = x_target.to(self.device)
            is_perturbed = torch.tensor([m['is_perturbed'] for m in metadata], 
                                         dtype=torch.float32).to(self.device)
            
            self.optimizer.zero_grad()
            
            t_span = torch.arange(x_target.size(1)).float().to(self.device)
            x_recon, x_pred, perturbation_prob, z_seq = self.model(x_sequence, t_span)
            
            losses = self.compute_regularized_loss(
                x_recon, x_pred, x_sequence, x_target,
                perturbation_prob, is_perturbed, z_seq, epoch
            )
            
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            
            self.optimizer.step()
            total_loss += losses['total'].item()
            
            # Track learning rate
            if batch_idx == 0:
                self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x_sequence, x_target, metadata in val_loader:
                x_sequence = x_sequence.to(self.device)
                x_target = x_target.to(self.device)
                is_perturbed = torch.tensor([m['is_perturbed'] for m in metadata], 
                                             dtype=torch.float32).to(self.device)
                
                t_span = torch.arange(x_target.size(1)).float().to(self.device)
                x_recon, x_pred, perturbation_prob, z_seq = self.model(x_sequence, t_span)
                
                losses = self.compute_regularized_loss(
                    x_recon, x_pred, x_sequence, x_target,
                    perturbation_prob, is_perturbed, z_seq, epoch
                )
                
                total_loss += losses['total'].item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, max_epochs=100):
        print("Starting anti-overfitting training...")
        
        for epoch in range(max_epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            val_ratio = val_loss / train_loss if train_loss > 0 else float('inf')
            self.val_ratios.append(val_ratio)
            
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, Ratio={val_ratio:.2f}")
            
            # Enhanced early stopping
            improvement = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                improvement = True
            
            if val_ratio < self.best_ratio:
                self.best_ratio = val_ratio
                improvement = True
            
            if improvement:
                self.patience_counter = 0
                # Save model
                os.makedirs(MODELS_CHECKPOINT_DIR, exist_ok=True)
                torch.save({ # <--- CORRECTED: Use global absolute path variable
                 'epoch': epoch,
                 'model_state_dict': self.model.state_dict(),
                 'train_loss': train_loss,
                 'val_loss': val_loss,
                             'val_ratio': val_ratio,
                }, os.path.join(MODELS_CHECKPOINT_DIR, 'anti_overfitting_best.pth'))
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        self.plot_results()
        return self.best_ratio
    
    def plot_results(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0,0].plot(self.train_losses, label='Train', alpha=0.8)
        axes[0,0].plot(self.val_losses, label='Validation', alpha=0.8)
        axes[0,0].set_yscale('log')
        axes[0,0].set_title('Loss Curves (Log Scale)')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Validation ratio
        axes[0,1].plot(self.val_ratios, color='red', alpha=0.8)
        axes[0,1].axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Perfect (1.0)')
        axes[0,1].axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='Good (<2.0)')
        axes[0,1].set_title('Validation/Training Ratio')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Loss difference
        loss_diff = [v - t for v, t in zip(self.val_losses, self.train_losses)]
        axes[1,0].plot(loss_diff, color='purple', alpha=0.8)
        axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1,0].set_title('Validation - Training Loss')
        axes[1,0].grid(True)
        
        # Learning rate
        axes[1,1].plot(self.learning_rates, color='green', alpha=0.8)
        axes[1,1].set_yscale('log')
        axes[1,1].set_title('Learning Rate Schedule')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_CHECKPOINT_DIR, 'anti_overfitting_results.png'), dpi=300) 
        plt.show() # Keep this line


def custom_collate_fn(batch):
    """Custom collate function for DataLoader."""
    sequences = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    metadata_list = [item[2] for item in batch]
    
    collated_sequences = default_collate(sequences)
    collated_targets = default_collate(targets)
    
    return collated_sequences, collated_targets, metadata_list


def run_enhanced_overfitting_solution():
    """Main solution pipeline with enhanced anti-overfitting measures."""
    
    print("🎯 ENHANCED OVERFITTING SOLUTION")
    print("=" * 50)
    
    # Data directory
    data_dir = DATA_ROOT_DIR 
    
    # Collect all data files
    all_files = []
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.npz'):
                all_files.append(os.path.join(root, filename))
    
    if len(all_files) == 0:
        print("❌ No data files found!")
        return
    
    print(f"Found {len(all_files)} data files")
    
    # Split into train/validation
    train_files, val_files = train_test_split(all_files, test_size=0.3, random_state=42)
    
    # Create enhanced datasets
    print("Creating enhanced training dataset...")
    train_dataset = ImprovedDataset(
        train_files,
        sequence_length=10,
        prediction_horizon=2,
        fit_scaler=True,
        augment_data=True,
        balance_classes=True
    )
    
    print("Creating validation dataset...")
    val_dataset = ImprovedDataset(
        val_files,
        sequence_length=10,
        prediction_horizon=2,
        scaler=train_dataset.scaler,
        fit_scaler=False,
        augment_data=False,
        balance_classes=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders with smaller batch size
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True,
        collate_fn=custom_collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        collate_fn=custom_collate_fn, pin_memory=True
    )
    
    # Create ultra-minimal model
    model = UltraMinimalNeuralODE(
        state_dim=18,
        latent_dim=3,
        hidden_dim=12,
        dropout_rate=0.5
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Train with anti-overfitting measures
    trainer = AntiOverfittingTrainer(model)
    final_ratio = trainer.train(train_loader, val_loader, max_epochs=80)
    
    print(f"\n🏁 FINAL RESULTS:")
    print(f"Best validation/training ratio: {final_ratio:.2f}")
    
    if final_ratio < 1.5:
        print("✅ EXCELLENT: Overfitting completely solved!")
    elif final_ratio < 2.0:
        print("✅ GOOD: Overfitting significantly reduced!")
    elif final_ratio < 2.5:
        print("⚠️  IMPROVED: Some progress, but may need more data diversity")
    else:
        print("❌ ISSUE: Still overfitting. Consider generating more diverse data.")
    
    return final_ratio


if __name__ == "__main__":
    # Run the enhanced solution
    run_enhanced_overfitting_solution()