# three_body_causal_ai/model/neural_ode_model_improved.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, default_collate
import numpy as np
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# --- ABSOLUTE PATH HANDLING ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
DATA_ROOT_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'model', 'checkpoints')

class ThreeBodyDataset(Dataset):
    """Dataset class for three-body simulation data with normalization."""
    
    def __init__(self, data_files, sequence_length=50, prediction_horizon=10, scaler=None, fit_scaler=False):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.sequences = []
        self.targets = []
        self.metadata = []
        self.scaler = scaler
        
        # Collect all data first
        all_data = []
        for file_path in data_files:
            data = self._load_file_data(file_path)
            if data is not None:
                all_data.extend(data)
        
        # Fit scaler if needed
        if fit_scaler and self.scaler is None:
            self.scaler = StandardScaler()
            all_sequences = np.array([item[0] for item in all_data])
            all_targets = np.array([item[1] for item in all_data])
            
            # Reshape for fitting scaler
            seq_shape = all_sequences.shape
            target_shape = all_targets.shape
            
            combined_data = np.concatenate([
                all_sequences.reshape(-1, seq_shape[-1]),
                all_targets.reshape(-1, target_shape[-1])
            ])
            
            self.scaler.fit(combined_data)
            print(f"Fitted scaler with mean: {self.scaler.mean_[:6]}")
            print(f"Fitted scaler with std: {self.scaler.scale_[:6]}")
        
        # Process and normalize data
        for sequence, target, metadata in all_data:
            if self.scaler is not None:
                # Normalize sequences and targets
                seq_shape = sequence.shape
                target_shape = target.shape
                
                sequence_norm = self.scaler.transform(sequence.reshape(-1, seq_shape[-1]))
                target_norm = self.scaler.transform(target.reshape(-1, target_shape[-1]))
                
                sequence = sequence_norm.reshape(seq_shape)
                target = target_norm.reshape(target_shape)
            
            self.sequences.append(sequence)
            self.targets.append(target)
            self.metadata.append(metadata)
    
    def _load_file_data(self, file_path):
        """Load and process a single .npz file."""
        try:
            data = np.load(file_path, allow_pickle=True)
            solution = data['solution']
            
            raw_perturbation_type = data.get('perturbation_type', None)
            perturbation_type_str = str(raw_perturbation_type) if raw_perturbation_type is not None else 'None'
            is_perturbed = 1 if perturbation_type_str != 'None' else 0
            
            file_data = []
            for i in range(len(solution) - self.sequence_length - self.prediction_horizon + 1):
                sequence = solution[i:i + self.sequence_length]
                target = solution[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
                
                metadata = {
                    'is_perturbed': is_perturbed,
                    'perturbation_type': perturbation_type_str,
                    'file': file_path
                }
                
                file_data.append((sequence, target, metadata))
            
            data.close()
            return file_data
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor(self.targets[idx])
        metadata = self.metadata[idx]
        
        return sequence, target, metadata

def custom_collate_fn(batch):
    sequences = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    metadata_list = [item[2] for item in batch]

    collated_sequences = default_collate(sequences)
    collated_targets = default_collate(targets)
    
    return collated_sequences, collated_targets, metadata_list

# --- Improved Neural ODE Function with Regularization ---
class LatentODEFunc(nn.Module):
    """Neural ODE function in latent space with dropout and weight decay."""
    
    def __init__(self, latent_dim=16, hidden_dim=64, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
    
    def forward(self, t, z):
        return self.net(z)

# --- Improved VAE Components ---
class VariationalEncoder(nn.Module):
    """Encoder for VAE component with batch normalization and dropout."""
    
    def __init__(self, input_dim, latent_dim=16, hidden_dim=128, dropout_rate=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, -10, 10)
        return mu, logvar

class Decoder(nn.Module):
    """Decoder for VAE component with batch normalization and dropout."""
    
    def __init__(self, latent_dim=16, output_dim=18, hidden_dim=128, dropout_rate=0.2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z):
        return self.decoder(z)

# --- Improved Neural ODE VAE Model ---
class ImprovedNeuralODEVAE(nn.Module):
    """Improved Neural ODE + VAE model with better regularization."""
    
    def __init__(self, state_dim=18, latent_dim=8, hidden_dim=64, dropout_rate=0.15):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        # Smaller, more regularized VAE components
        self.encoder = VariationalEncoder(state_dim, latent_dim, hidden_dim, dropout_rate)
        self.decoder = Decoder(latent_dim, state_dim, hidden_dim, dropout_rate)
        
        # Smaller Neural ODE function
        self.ode_func = LatentODEFunc(latent_dim, hidden_dim // 2, dropout_rate)
        
        # Simpler perturbation classifier
        self.perturbation_classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        """VAE reparameterization trick with improved numerical stability."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # Use mean during inference
    
    def encode(self, x):
        """Encode states to latent space."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        """Decode from latent space to state space."""
        return self.decoder(z)
    
    def forward(self, x_sequence, t_span):
        """Forward pass with improved stability."""
        batch_size, seq_len, state_dim = x_sequence.shape
        
        # Encode entire sequence
        x_flat = x_sequence.reshape(-1, state_dim)
        z_flat, mu, logvar = self.encode(x_flat)
        z_seq = z_flat.reshape(batch_size, seq_len, self.latent_dim)
        
        # Use last latent state as initial condition
        z0 = z_seq[:, -1, :]
        
        # Integrate ODE with error handling
        try:
            z_pred = odeint(self.ode_func, z0, t_span, method='rk4', 
                          options={'step_size': 0.1})
            z_pred = z_pred.permute(1, 0, 2)
        except Exception as e:
            print(f"ODE integration failed: {e}")
            # Fallback: simple linear extrapolation
            z_pred = z0.unsqueeze(1).repeat(1, len(t_span), 1)
        
        # Decode predictions
        z_pred_flat = z_pred.reshape(-1, self.latent_dim)
        x_pred_flat = self.decode(z_pred_flat)
        x_pred = x_pred_flat.reshape(batch_size, len(t_span), state_dim)
        
        # Reconstruct input sequence
        x_recon_flat = self.decode(z_flat)
        x_recon = x_recon_flat.reshape(batch_size, seq_len, state_dim)
        
        # Perturbation detection
        z_mean_for_classifier = torch.mean(z_seq, dim=1)
        perturbation_prob = self.perturbation_classifier(z_mean_for_classifier)
        
        return x_recon, x_pred, perturbation_prob, mu, logvar, z_seq

# --- Improved Trainer with Better Regularization ---
class ImprovedThreeBodyTrainer:
    """Training class with better overfitting prevention."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 learning_rate=5e-4, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        
        # Lower learning rate and weight decay for regularization
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                                   weight_decay=weight_decay)
        
        # More aggressive learning rate scheduling
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, 
            min_lr=1e-6
        )
        
        # Early stopping
        self.early_stopping_patience = 15
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        
        self.train_losses = []
        self.val_losses = []
    
    def compute_loss(self, x_recon, x_pred, x_sequence, x_target, 
                     perturbation_prob, is_perturbed, mu, logvar):
        """Compute loss with better balancing and regularization."""
        
        # Reconstruction loss
        recon_loss = nn.MSELoss()(x_recon, x_sequence)
        
        # Prediction loss
        pred_loss = nn.MSELoss()(x_pred, x_target)
        
        # KL divergence with beta scheduling
        beta = min(1.0, len(self.train_losses) / 100.0)  # Gradually increase beta
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
        
        # Classification loss
        perturbation_target = is_perturbed.float().unsqueeze(1)
        class_loss = nn.BCELoss()(perturbation_prob, perturbation_target)
        
        # Rebalanced loss weights - reduced prediction weight
        total_loss = (1.0 * recon_loss + 1.0 * pred_loss + 
                      beta * 0.1 * kl_loss + 0.3 * class_loss)
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'prediction': pred_loss,
            'kl_divergence': kl_loss,
            'classification': class_loss
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch with gradient clipping."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (x_sequence, x_target, metadata) in enumerate(train_loader):
            x_sequence = x_sequence.to(self.device)
            x_target = x_target.to(self.device)
            is_perturbed = torch.tensor([m['is_perturbed'] for m in metadata], 
                                      dtype=torch.float32).to(self.device)
            
            self.optimizer.zero_grad()
            
            t_span = torch.arange(x_target.size(1)).float().to(self.device)
            x_recon, x_pred, perturbation_prob, mu, logvar, z_seq = self.model(
                x_sequence, t_span
            )
            
            losses = self.compute_loss(
                x_recon, x_pred, x_sequence, x_target,
                perturbation_prob, is_perturbed, mu, logvar
            )
            
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            
            self.optimizer.step()
            total_loss += losses['total'].item()
            
            if batch_idx % 20 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}: Total Loss = {losses["total"].item():.6f}')
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x_sequence, x_target, metadata in val_loader:
                x_sequence = x_sequence.to(self.device)
                x_target = x_target.to(self.device)
                is_perturbed = torch.tensor([m['is_perturbed'] for m in metadata], 
                                          dtype=torch.float32).to(self.device)
                
                t_span = torch.arange(x_target.size(1)).float().to(self.device)
                x_recon, x_pred, perturbation_prob, mu, logvar, z_seq = self.model(
                    x_sequence, t_span
                )
                
                losses = self.compute_loss(
                    x_recon, x_pred, x_sequence, x_target,
                    perturbation_prob, is_perturbed, mu, logvar
                )
                
                total_loss += losses['total'].item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=100, save_dir=MODELS_CHECKPOINT_DIR):
        """Main training loop with early stopping."""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Training Improved Neural ODE VAE for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1} Summary - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model_improved.pth'))
                print(f"New best model saved! Val Loss: {val_loss:.6f}")
            else:
                self.early_stopping_counter += 1
                print(f"Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print("Early stopping triggered!")
                    break
        
        self.plot_training_history(save_dir)
        print("Training completed!")
    
    def plot_training_history(self, save_dir):
        """Plot training and validation loss."""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, label='Training Loss', alpha=0.8)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # Log scale to better see the trends
        
        plt.subplot(2, 1, 2)
        # Plot validation/training loss ratio to visualize overfitting
        if len(self.train_losses) > 0:
            ratios = [v/t if t > 0 else 1 for v, t in zip(self.val_losses, self.train_losses)]
            plt.plot(ratios, label='Val/Train Loss Ratio', color='red')
            plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Perfect Fit')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Ratio')
            plt.title('Overfitting Monitor (Lower is Better)')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'improved_training_history.png'))
        plt.show()

# --- Data Preparation with Normalization ---
def prepare_improved_data(data_dir=DATA_ROOT_DIR, test_size=0.2):
    """Prepare training and validation datasets with normalization."""
    
    all_files = []
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.npz'):
                all_files.append(os.path.join(root, filename))
    
    print(f"Found {len(all_files)} data files")
    
    # Split files
    train_files, val_files = train_test_split(all_files, test_size=test_size, random_state=42)
    
    # Create datasets with normalization
    print("Creating training dataset and fitting scaler...")
    train_dataset = ThreeBodyDataset(train_files, sequence_length=20, prediction_horizon=5,
                                   fit_scaler=True)  # Shorter sequences to reduce overfitting
    
    print("Creating validation dataset with fitted scaler...")
    val_dataset = ThreeBodyDataset(val_files, sequence_length=20, prediction_horizon=5,
                                 scaler=train_dataset.scaler)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Save scaler for later use
    scaler_path = os.path.join(MODELS_CHECKPOINT_DIR, 'data_scaler.pkl')
    os.makedirs(MODELS_CHECKPOINT_DIR, exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(train_dataset.scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    # Smaller batch size to reduce overfitting
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                            num_workers=0, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, 
                          num_workers=0, collate_fn=custom_collate_fn)
    
    return train_loader, val_loader, train_dataset, val_dataset

def main():
    """Main training script with overfitting prevention."""
    print("=== IMPROVED NEURAL ODE VAE TRAINING (OVERFITTING PREVENTION) ===")
    
    # Prepare data with normalization
    train_loader, val_loader, train_dataset, val_dataset = prepare_improved_data(
        data_dir=DATA_ROOT_DIR
    )
    
    if len(train_dataset) == 0:
        print("No training data found! Please ensure data is generated and paths are correct.")
        return
    
    # Initialize smaller, more regularized model
    state_dim = 18
    latent_dim = 8  # Reduced from 16
    hidden_dim = 64  # Reduced from 128
    model = ImprovedNeuralODEVAE(state_dim=state_dim, latent_dim=latent_dim, 
                               hidden_dim=hidden_dim, dropout_rate=0.15)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer with lower learning rate
    trainer = ImprovedThreeBodyTrainer(model, learning_rate=3e-4, weight_decay=1e-4)
    
    # Train model with early stopping
    trainer.train(train_loader, val_loader, epochs=100, 
                 save_dir=MODELS_CHECKPOINT_DIR)
    
    print("Improved training complete!")
    print("Key improvements:")
    print("- Smaller model architecture")
    print("- Data normalization")
    print("- Dropout and batch normalization")
    print("- Weight decay regularization") 
    print("- Early stopping")
    print("- Gradient clipping")
    print("- Better loss balancing")

if __name__ == '__main__':
    main()