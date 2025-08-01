# three_body_causal_ai/causal/causal_discovery.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler # For loading the scaler
from sklearn.decomposition import PCA
import networkx as nx
from scipy import stats # For t-tests
from scipy.stats import pearsonr # For correlation in graph building
import torch
import os
import warnings
import pickle # For loading scaler

# --- Enhanced 3D plotting imports ---
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
# --- End 3D plotting imports ---

# Suppress warnings for cleaner output (e.g., from statsmodels for Granger)
warnings.filterwarnings('ignore')

# --- Path Setup (CRITICAL: These must be robustly defined) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..') # Goes from 'causal' to 'three_body_causal_ai'
DATA_ROOT_DIR = os.path.join(PROJECT_ROOT, 'data') # Points to 'three_body_causal_ai/data'
MODELS_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'model', 'checkpoints') # Points to 'three_body_causal_ai/model/checkpoints'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results') # Points to 'three_body_causal_ai/results'

# --- Import your trained model and dataset utilities from overfitting_solution.py ---
import sys
sys.path.append(os.path.join(PROJECT_ROOT, 'model')) # Add 'model' directory to Python path
# Import the specific classes you trained and used in overfitting_solution.py
from overfitting_solution import UltraMinimalNeuralODE, ImprovedDataset # Use ImprovedDataset's loading methods

# --- Causal Discovery Libraries (Optional: install these if you want to use them) ---
try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
    print("Info: DoWhy available for advanced causal inference.")
except ImportError:
    print("Warning: DoWhy not available. Install with: pip install dowhy")
    DOWHY_AVAILABLE = False

try:
    # causalnex might have Python version compatibility issues.
    # We will proceed without it if it's not installed.
    from causalnex.structure import StructureModel
    from causalnex.structure.notears import from_pandas
    CAUSALNEX_AVAILABLE = True
    print("Info: CausalNex available for structural causal modeling.")
except ImportError:
    print("Warning: CausalNex not available. Install with: pip install causalnex (may require specific Python/OS setup, often incompatible with Python 3.11+).")
    CAUSALNEX_AVAILABLE = False

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    STATSMODELS_AVAILABLE = True
    print("Info: Statsmodels (Granger Causality) available.")
except ImportError:
    print("Warning: Statsmodels not available. Install with: pip install statsmodels")
    STATSMODELS_AVAILABLE = False


class LatentSpaceAnalyzer:
    """Analyze latent representations from trained Neural ODE VAE."""
    
    def __init__(self, model_path, scaler_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.scaler = None
        # These will store segmented data for causal analysis (one entry per sequence)
        self.latent_representations_segments = [] 
        self.metadata_for_causal_segments = []
        # This will store full latent sequences per original file for Granger (if needed)
        self.full_latent_sequences_per_file = {}

        # Load model and scaler during initialization
        self.load_model(model_path)
        self.load_scaler(scaler_path)
    
    def load_model(self, model_path):
        """Load trained UltraMinimalNeuralODE model."""
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
        
        # Instantiate model with the exact parameters it was trained with
        # These MUST match UltraMinimalNeuralODE in overfitting_solution.py
        model_params = {
            'state_dim': 18,
            'latent_dim': 3,  # Correct latent_dim
            'hidden_dim': 12, # Correct hidden_dim
            'dropout_rate': 0.5 # Correct dropout_rate
        }
        self.model = UltraMinimalNeuralODE(**model_params).to(self.device)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval() # Set model to evaluation mode
            print(f"Model loaded successfully from {model_path}! (Epoch {checkpoint['epoch']}, Val Loss {checkpoint['val_loss']:.4f}, Ratio {checkpoint['val_ratio']:.2f})")
        except Exception as e:
            print(f"Error loading model state_dict from {model_path}: {e}")
            self.model = None # Ensure model is None if loading fails

    def load_scaler(self, scaler_path):
        """Load the RobustScaler used for data normalization."""
        if not os.path.exists(scaler_path):
            print(f"Error: Scaler file not found at {scaler_path}. Please ensure it was saved during training.")
            return
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Scaler loaded successfully from {scaler_path}.")
        except Exception as e:
            print(f"Error loading scaler from {scaler_path}: {e}")
            self.scaler = None

    def extract_latent_representations(self, data_files, sequence_length=10, prediction_horizon=2):
        """
        Extract latent representations for all data files, segmenting them into
        sequences consistent with training, and storing for causal analysis.
        """
        if self.model is None or self.scaler is None:
            print("Model or Scaler not loaded! Cannot extract latent representations.")
            return
        
        self.latent_representations_segments = []
        self.metadata_for_causal_segments = []
        self.full_latent_sequences_per_file = {} # Clear previous runs

        # Replicate segmentation and loading logic from ImprovedDataset
        # Create a dummy dataset instance to use its _load_and_segment_data and _normalize_data methods
        temp_dataset_loader = ImprovedDataset([], sequence_length=sequence_length, 
                                            prediction_horizon=prediction_horizon, 
                                            scaler=self.scaler, fit_scaler=False, 
                                            augment_data=False, balance_classes=False)
        
        print(f"\nExtracting latent representations from {len(data_files)} files...")
        
        for file_path in data_files:
            # Load the full solution data from the .npz file
            try:
                with np.load(file_path, allow_pickle=True) as data_npz:
                    full_solution_unnorm = data_npz['solution'] # This is the original, unnormalized full trajectory
                    
                # Normalize the full solution for model input
                full_solution_norm = self.scaler.transform(full_solution_unnorm)

                # Get segments from this full solution for feature extraction
                clean_samples_in_file, perturbed_samples_in_file = temp_dataset_loader._load_and_segment_data([file_path])
                all_segments_in_file = clean_samples_in_file + perturbed_samples_in_file
                
                # --- Get full latent sequence for Granger Causality ---
                # Pass the full normalized solution through the model to get its full latent sequence
                x_full_tensor = torch.FloatTensor(full_solution_norm).unsqueeze(0).to(self.device) # Add batch dim
                # Dummy t_span is needed but its length doesn't affect z_seq output from forward pass for full trajectory
                dummy_t_span = torch.arange(full_solution_norm.shape[0]).float().to(self.device)
                
                with torch.no_grad():
                    # model.forward() returns (x_recon, x_pred, perturbation_prob, z_seq)
                    _, _, _, z_seq_full_tensor = self.model(x_full_tensor, dummy_t_span)
                
                self.full_latent_sequences_per_file[file_path] = z_seq_full_tensor.squeeze(0).cpu().numpy()
                # --- End Granger preparation ---

                # --- Process each segment for aggregated features ---
                for seq_orig, target_orig, meta_orig in all_segments_in_file:
                    # Normalize the segment (already done implicitly by _load_and_segment_data)
                    seq_norm_segment = self.scaler.transform(seq_orig) # Ensure this is normalized
                    
                    x_segment_tensor = torch.FloatTensor(seq_norm_segment).unsqueeze(0).to(self.device)
                    t_span_segment = torch.arange(prediction_horizon).float().to(self.device) # Prediction horizon for segment
                    
                    with torch.no_grad():
                        _, _, perturbation_prob_segment, z_seq_segment_tensor = self.model(x_segment_tensor, t_span_segment)
                    
                    # Store results for this segment
                    self.latent_representations_segments.append({
                        'latent_sequence_segment': z_seq_segment_tensor.cpu().numpy().squeeze(0), # (seq_len, latent_dim)
                        'latent_mean': torch.mean(z_seq_segment_tensor, dim=1).cpu().numpy().flatten(), # (latent_dim,)
                        'perturbation_prob_model': perturbation_prob_segment.item(), # Scalar probability for this segment
                        'original_data_segment_unnorm': seq_orig # Store original UN-NORMALIZED segment
                    })
                    
                    # Store metadata (combine original file metadata with segment-specific info)
                    self.metadata_for_causal_segments.append(meta_orig.copy()) # Copy to avoid modifying original
                    
                print(f"Processed: {os.path.basename(file_path)} ({len(all_segments_in_file)} segments extracted)")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

    def _create_custom_colormap(self):
        """Create a beautiful custom colormap for 3D visualizations."""
        colors = ['#0d1117', '#1f2937', '#374151', '#6b7280', '#9ca3af', 
                 '#d1d5db', '#f3f4f6', '#fef3c7', '#fbbf24', '#f59e0b', 
                 '#d97706', '#b45309', '#92400e', '#78350f']
        return mcolors.LinearSegmentedColormap.from_list("custom", colors)

    def _setup_3d_plot_style(self, ax, title):
        """Apply consistent styling to 3D plots."""
        # Set background colors
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Set pane edges
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        # Grid styling
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Axis styling
        ax.xaxis.label.set_color('darkslategray')
        ax.yaxis.label.set_color('darkslategray')
        ax.zaxis.label.set_color('darkslategray')
        
        # Title styling
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20, color='darkslategray')
        
        return ax

    def visualize_latent_space(self, save_dir=os.path.join(RESULTS_DIR, 'latent_analysis')):
        """Enhanced visualization with improved 3D plots and multiple views."""
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.latent_representations_segments:
            print("No latent representations available for visualization!")
            return None, None
            
        latent_means_for_plot = np.array([lr['latent_mean'] for lr in self.latent_representations_segments])
        labels = np.array([meta['is_perturbed'] for meta in self.metadata_for_causal_segments])
        perturbation_types = np.array([meta['perturbation_type'] for meta in self.metadata_for_causal_segments])
        perturbation_probs = np.array([lr['perturbation_prob_model'] for lr in self.latent_representations_segments])
        
        if latent_means_for_plot.shape[0] == 0:
            print("No data points for visualization.")
            return None, None

        # --- Enhanced 2D PCA (keeping original functionality) ---
        pca = PCA(n_components=2)
        latent_2d_pca = pca.fit_transform(latent_means_for_plot)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

        fig_2d_pca, axes_2d_pca = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1 (2D PCA): Clean vs Perturbed (True Label)
        colors_binary = ['#2563eb', '#dc2626']  # Modern blue and red
        labels_binary = ['Clean', 'Perturbed']
        for i, label in enumerate(labels_binary):
            mask = labels == i
            if np.any(mask):
                axes_2d_pca[0].scatter(latent_2d_pca[mask, 0], latent_2d_pca[mask, 1], 
                                c=colors_binary[i], label=label, alpha=0.7, s=25, edgecolors='white', linewidth=0.5)
        axes_2d_pca[0].set_title('Latent Space (2D PCA): True Clean vs Perturbed', fontweight='bold')
        axes_2d_pca[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)', fontweight='bold')
        axes_2d_pca[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)', fontweight='bold')
        axes_2d_pca[0].legend(frameon=True, fancybox=True, shadow=True)
        axes_2d_pca[0].grid(True, alpha=0.3, linestyle='--')

        # Plot 2 (2D PCA): By Specific Perturbation Type
        unique_types = sorted(list(set(perturbation_types)))
        colors_discrete = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
        for i, ptype in enumerate(unique_types):
            mask = perturbation_types == ptype
            if np.any(mask):
                axes_2d_pca[1].scatter(latent_2d_pca[mask, 0], latent_2d_pca[mask, 1], 
                                c=[colors_discrete[i]], label=ptype, alpha=0.7, s=25, 
                                edgecolors='white', linewidth=0.5)
        axes_2d_pca[1].set_title('Latent Space (2D PCA): By Perturbation Type', fontweight='bold')
        axes_2d_pca[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)', fontweight='bold')
        axes_2d_pca[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)', fontweight='bold')
        axes_2d_pca[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., 
                             frameon=True, fancybox=True, shadow=True)
        axes_2d_pca[1].grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(os.path.join(save_dir, 'latent_space_overview_2d.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig_2d_pca)

        # --- ENHANCED 3D Plotting Suite ---
        
        # 3D Plot 1: Clean vs Perturbed with enhanced styling
        fig_3d_binary = plt.figure(figsize=(12, 9))
        ax_3d_binary = fig_3d_binary.add_subplot(111, projection='3d')
        ax_3d_binary = self._setup_3d_plot_style(ax_3d_binary, 'Latent Space (3D): Clean vs Perturbed')

        for i, label in enumerate(labels_binary):
            mask = labels == i
            if np.any(mask):
                scatter = ax_3d_binary.scatter(latent_means_for_plot[mask, 0], 
                                             latent_means_for_plot[mask, 1], 
                                             latent_means_for_plot[mask, 2],
                                             c=colors_binary[i], label=label, alpha=0.8, s=40,
                                             edgecolors='white', linewidth=0.5, 
                                             depthshade=True)
        
        ax_3d_binary.set_xlabel('Latent Dim 0', fontweight='bold', labelpad=10)
        ax_3d_binary.set_ylabel('Latent Dim 1', fontweight='bold', labelpad=10)
        ax_3d_binary.set_zlabel('Latent Dim 2', fontweight='bold', labelpad=10)
        ax_3d_binary.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        # Set viewing angle for better perspective
        ax_3d_binary.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'latent_space_3d_binary_enhanced.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig_3d_binary)

        # 3D Plot 2: By Perturbation Type with enhanced styling
        fig_3d_types = plt.figure(figsize=(14, 10))
        ax_3d_types = fig_3d_types.add_subplot(111, projection='3d')
        ax_3d_types = self._setup_3d_plot_style(ax_3d_types, 'Latent Space (3D): By Perturbation Type')

        for i, ptype in enumerate(unique_types):
            mask = perturbation_types == ptype
            if np.any(mask):
                scatter = ax_3d_types.scatter(latent_means_for_plot[mask, 0], 
                                            latent_means_for_plot[mask, 1], 
                                            latent_means_for_plot[mask, 2],
                                            c=[colors_discrete[i]], label=ptype, alpha=0.8, s=40,
                                            edgecolors='white', linewidth=0.5, 
                                            depthshade=True)
        
        ax_3d_types.set_xlabel('Latent Dim 0', fontweight='bold', labelpad=10)
        ax_3d_types.set_ylabel('Latent Dim 1', fontweight='bold', labelpad=10)
        ax_3d_types.set_zlabel('Latent Dim 2', fontweight='bold', labelpad=10)
        ax_3d_types.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0.,
                          frameon=True, fancybox=True, shadow=True)
        
        # Set viewing angle for better perspective
        ax_3d_types.view_init(elev=25, azim=60)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(os.path.join(save_dir, 'latent_space_3d_types_enhanced.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig_3d_types)

        # 3D Plot 3: NEW - Colored by Model Perturbation Probability (Continuous)
        fig_3d_prob = plt.figure(figsize=(12, 9))
        ax_3d_prob = fig_3d_prob.add_subplot(111, projection='3d')
        ax_3d_prob = self._setup_3d_plot_style(ax_3d_prob, 'Latent Space (3D): Colored by Model Perturbation Probability')

        # Create a color map based on perturbation probabilities
        scatter_prob = ax_3d_prob.scatter(latent_means_for_plot[:, 0], 
                                        latent_means_for_plot[:, 1], 
                                        latent_means_for_plot[:, 2],
                                        c=perturbation_probs, cmap='plasma', alpha=0.8, s=50,
                                        edgecolors='white', linewidth=0.5, 
                                        depthshade=True)
        
        # Add colorbar
        cbar = plt.colorbar(scatter_prob, ax=ax_3d_prob, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('Model Perturbation Probability', fontweight='bold', labelpad=15)
        cbar.ax.tick_params(labelsize=10)
        
        ax_3d_prob.set_xlabel('Latent Dim 0', fontweight='bold', labelpad=10)
        ax_3d_prob.set_ylabel('Latent Dim 1', fontweight='bold', labelpad=10)
        ax_3d_prob.set_zlabel('Latent Dim 2', fontweight='bold', labelpad=10)
        
        # Set viewing angle for better perspective
        ax_3d_prob.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'latent_space_3d_probability_enhanced.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig_3d_prob)

        # 3D Plot 4: NEW - Multiple Views in One Figure
        fig_multi = plt.figure(figsize=(20, 6))
        
        # View 1: Front view
        ax1 = fig_multi.add_subplot(131, projection='3d')
        ax1 = self._setup_3d_plot_style(ax1, 'Front View (elev=0°, azim=0°)')
        for i, label in enumerate(labels_binary):
            mask = labels == i
            if np.any(mask):
                ax1.scatter(latent_means_for_plot[mask, 0], latent_means_for_plot[mask, 1], 
                           latent_means_for_plot[mask, 2], c=colors_binary[i], label=label, 
                           alpha=0.8, s=30, edgecolors='white', linewidth=0.5)
        ax1.view_init(elev=0, azim=0)
        ax1.legend(loc='upper right', fontsize=9)
        
        # View 2: Side view
        ax2 = fig_multi.add_subplot(132, projection='3d')
        ax2 = self._setup_3d_plot_style(ax2, 'Side View (elev=0°, azim=90°)')
        for i, label in enumerate(labels_binary):
            mask = labels == i
            if np.any(mask):
                ax2.scatter(latent_means_for_plot[mask, 0], latent_means_for_plot[mask, 1], 
                           latent_means_for_plot[mask, 2], c=colors_binary[i], label=label, 
                           alpha=0.8, s=30, edgecolors='white', linewidth=0.5)
        ax2.view_init(elev=0, azim=90)
        ax2.legend(loc='upper right', fontsize=9)
        
        # View 3: Isometric view
        ax3 = fig_multi.add_subplot(133, projection='3d')
        ax3 = self._setup_3d_plot_style(ax3, 'Isometric View (elev=30°, azim=45°)')
        for i, label in enumerate(labels_binary):
            mask = labels == i
            if np.any(mask):
                ax3.scatter(latent_means_for_plot[mask, 0], latent_means_for_plot[mask, 1], 
                           latent_means_for_plot[mask, 2], c=colors_binary[i], label=label, 
                           alpha=0.8, s=30, edgecolors='white', linewidth=0.5)
        ax3.view_init(elev=30, azim=45)
        ax3.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'latent_space_3d_multi_view.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig_multi)

        print("Enhanced 3D visualizations completed!")
        return pca, latent_2d_pca

    def create_interactive_3d_trajectory_visualization(self, save_dir=os.path.join(RESULTS_DIR, 'latent_analysis')):
        """Create 3D trajectory visualization showing latent space evolution over time."""
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.latent_representations_segments:
            print("No latent representations available for trajectory visualization!")
            return
        
        # Select a few representative sequences for trajectory visualization
        n_trajectories = min(10, len(self.latent_representations_segments))
        selected_indices = np.linspace(0, len(self.latent_representations_segments)-1, n_trajectories, dtype=int)
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax = self._setup_3d_plot_style(ax, 'Latent Space Trajectories (Time Evolution)')
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))
        
        for i, idx in enumerate(selected_indices):
            latent_seq = self.latent_representations_segments[idx]['latent_sequence_segment']
            meta = self.metadata_for_causal_segments[idx]
            
            if latent_seq.shape[0] > 1:  # Only plot if we have a sequence
                # Plot trajectory line
                ax.plot(latent_seq[:, 0], latent_seq[:, 1], latent_seq[:, 2], 
                       color=colors[i], alpha=0.7, linewidth=2, 
                       label=f"{meta['perturbation_type'][:8]}...")
                
                # Mark start and end points
                ax.scatter(latent_seq[0, 0], latent_seq[0, 1], latent_seq[0, 2], 
                          color=colors[i], s=100, marker='o', alpha=0.9,
                          edgecolors='white', linewidth=2)  # Start point
                ax.scatter(latent_seq[-1, 0], latent_seq[-1, 1], latent_seq[-1, 2], 
                          color=colors[i], s=100, marker='s', alpha=0.9,
                          edgecolors='white', linewidth=2)  # End point
        
        ax.set_xlabel('Latent Dim 0', fontweight='bold', labelpad=10)
        ax.set_ylabel('Latent Dim 1', fontweight='bold', labelpad=10)
        ax.set_zlabel('Latent Dim 2', fontweight='bold', labelpad=10)
        
        # Create custom legend
        circle_patch = mpatches.Patch(color='gray', label='Trajectory Start (○)')
        square_patch = mpatches.Patch(color='gray', label='Trajectory End (□)')
        ax.legend(handles=[circle_patch, square_patch], loc='upper left', 
                 frameon=True, fancybox=True, shadow=True)
        
        ax.view_init(elev=25, azim=60)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'latent_space_3d_trajectories.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        print(f"3D trajectory visualization created with {n_trajectories} representative trajectories!")


# --- Enhanced Causal Graph Builder ---
class CausalGraphBuilder:
    """Build causal graphs from latent representations and physical features."""
    
    def __init__(self, latent_analyzer):
        self.latent_analyzer = latent_analyzer
        self.causal_data = None
        self.causal_graph = None
    
    def prepare_causal_dataset(self):
        """
        Prepare dataset for causal discovery by combining latent means,
        perturbation probabilities, and physical features.
        Each row corresponds to one (sequence_length, prediction_horizon) segment.
        """
        if not self.latent_analyzer.latent_representations_segments:
            print("No latent representations available for causal dataset preparation!")
            return None
            
        data_rows = []
        
        for i, lr_data in enumerate(self.latent_analyzer.latent_representations_segments):
            meta = self.latent_analyzer.metadata_for_causal_segments[i]
            
            latent_mean = lr_data['latent_mean'] # Already (latent_dim,)
            original_data_segment_unnorm = lr_data['original_data_segment_unnorm'] # Un-normalized segment
            
            # Compute physical features from the UN-NORMALIZED segment
            physical_features = self._compute_physical_features(original_data_segment_unnorm)
            
            row = {}
            for j in range(self.latent_analyzer.model.latent_dim): # Use actual latent_dim from model
                row[f'latent_{j}'] = latent_mean[j]
            
            row.update(physical_features)
            
            # Target variables and perturbation type flags
            row['is_perturbed'] = meta['is_perturbed']
            row['perturbation_prob_model'] = lr_data['perturbation_prob_model'] # Model's prediction for this segment
            
            perturbation_types_list = ['clean', 'hidden_mass', 'drag', 'non_inverse_square', 'time_varying_G', 'impulse']
            for ptype in perturbation_types_list:
                row[f'is_{ptype}'] = 1 if meta['perturbation_type'] == ptype else 0
                
            data_rows.append(row)
                
        self.causal_data = pd.DataFrame(data_rows)
        # Handle non-numeric columns if present after update, before correlation
        for col in self.causal_data.columns:
            if self.causal_data[col].dtype == 'object': # Check for object dtype
                try:
                    self.causal_data[col] = pd.to_numeric(self.causal_data[col], errors='coerce')
                except:
                    pass # Keep as object if cannot convert
        self.causal_data = self.causal_data.dropna(axis=1, how='all') # Drop columns that became all NaN
        print(f"Prepared causal dataset with {len(self.causal_data)} samples and {len(self.causal_data.columns)} features.")
        return self.causal_data
    
    def _compute_physical_features(self, solution_data_segment):
        """
        Compute interpretable physical features from a single UN-NORMALIZED trajectory segment.
        solution_data_segment shape: (sequence_length, state_dim)
        """
        n_bodies = 3 
        features = {}
        
        if solution_data_segment.shape[0] == 0: # Handle empty segments
            return {f'body{b}_avg_distance': 0.0 for b in range(n_bodies)} # Return dummy zero features
        
        for body in range(n_bodies):
            pos_coords = solution_data_segment[:, body*6 : body*6 + 3]
            vel_coords = solution_data_segment[:, body*6 + 3 : body*6 + 6]
            
            distances = np.linalg.norm(pos_coords, axis=1)
            features[f'body{body}_avg_distance'] = np.mean(distances)
            features[f'body{body}_distance_std'] = np.std(distances)
            features[f'body{body}_max_distance'] = np.max(distances)
            
            speeds = np.linalg.norm(vel_coords, axis=1)
            features[f'body{body}_avg_speed'] = np.mean(speeds)
            features[f'body{body}_speed_std'] = np.std(speeds)
            features[f'body{body}_max_speed'] = np.max(speeds)
            
            features[f'body{body}_pos_range_x'] = np.ptp(pos_coords[:, 0])
            features[f'body{body}_pos_range_y'] = np.ptp(pos_coords[:, 1])
            features[f'body{body}_pos_range_z'] = np.ptp(pos_coords[:, 2])
        
        # System-level features (assuming unit masses for simplicity as masses not passed with segments)
        total_ke_segment = 0.5 * np.sum(np.linalg.norm(solution_data_segment[:, 3:], axis=1)**2) # Sum of 0.5*v^2 for all 3 bodies across all axes
        
        features['system_avg_kinetic_energy'] = np.mean(total_ke_segment)
        features['system_kinetic_energy_std'] = np.std(total_ke_segment)
        
        # Pairwise distances
        for i in range(n_bodies):
            for j in range(i+1, n_bodies):
                pos_i = solution_data_segment[:, [i*6, i*6+1, i*6+2]]
                pos_j = solution_data_segment[:, [j*6, j*6+1, j*6+2]]
                pairwise_distances = np.linalg.norm(pos_i - pos_j, axis=1)
                
                features[f'pair_{i}_{j}_avg_distance'] = np.mean(pairwise_distances)
                features[f'pair_{i}_{j}_min_distance'] = np.min(pairwise_distances)
                features[f'pair_{i}_{j}_distance_std'] = np.std(pairwise_distances)
                
        return features
    
    def build_correlation_graph(self, threshold=0.3, save_dir=os.path.join(RESULTS_DIR, 'causal_analysis')):
        """Build causal graph based on correlations between all features."""
        os.makedirs(save_dir, exist_ok=True)
        
        if self.causal_data is None:
            print("No causal data available! Run prepare_causal_dataset first.")
            return None
            
        numeric_cols = self.causal_data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.causal_data[numeric_cols].corr()
        
        G = nx.Graph()
        
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:
                    corr_val = correlation_matrix.loc[col1, col2]
                    if abs(corr_val) > threshold:
                        G.add_edge(col1, col2, weight=abs(corr_val), correlation=corr_val)
        
        self.causal_graph = G
        self.visualize_causal_graph(save_dir) # Call visualization immediately after building
        return G
    
    def visualize_causal_graph(self, save_dir=os.path.join(RESULTS_DIR, 'causal_analysis')):
        """Enhanced 3D causal graph visualization with improved styling."""
        os.makedirs(save_dir, exist_ok=True)
        
        if self.causal_graph is None:
            print("No causal graph available!")
            return
        
        # Create both 2D and 3D visualizations
        self._create_2d_causal_graph(save_dir)
        self._create_3d_causal_graph(save_dir)
        
    def _create_2d_causal_graph(self, save_dir):
        """Create enhanced 2D causal graph visualization."""
        plt.figure(figsize=(20, 14))
        
        pos = nx.spring_layout(self.causal_graph, k=1.2, iterations=150, seed=42)
        
        node_colors = []
        node_sizes = []
        node_labels = {}
        
        for node in self.causal_graph.nodes():
            if 'is_' in node and ('perturbed' in node or '_type' in node):
                node_colors.append('#dc2626')  # Red for perturbation flags
                node_sizes.append(1200)
            elif 'latent_' in node:
                node_colors.append('#3b82f6')  # Blue for latent features
                node_sizes.append(800)
            elif 'prob' in node:
                node_colors.append('#f59e0b')  # Orange for probabilities
                node_sizes.append(900)
            elif 'body' in node:
                node_colors.append('#10b981')  # Green for body features
                node_sizes.append(600)
            elif 'system_' in node:
                node_colors.append('#8b5cf6')  # Purple for system features
                node_sizes.append(700)
            elif 'pair_' in node:
                node_colors.append('#f97316')  # Orange for pairwise features
                node_sizes.append(600)
            else:
                node_colors.append('#6b7280')  # Gray for others
                node_sizes.append(500)
            
            # Create abbreviated labels
            if len(node) > 15:
                abbreviated = node.replace('is_perturbed', 'Perturbed')\
                                .replace('perturbation_prob_model', 'PredProb')\
                                .replace('is_', '').replace('system_', 'Sys_')\
                                .replace('avg_', '').replace('distance', 'dist')\
                                .replace('kinetic_energy', 'KE')[:12] + '...'
                node_labels[node] = abbreviated
            else:
                node_labels[node] = node
        
        # Draw nodes with enhanced styling
        nx.draw_networkx_nodes(self.causal_graph, pos, 
                              node_color=node_colors, node_size=node_sizes, 
                              alpha=0.9, edgecolors='white', linewidths=2)
        
        # Draw edges with varying styles
        edge_colors = []
        edge_widths = []
        edge_styles = []
        
        for u, v, data in self.causal_graph.edges(data=True):
            corr = data['correlation']
            if corr > 0:
                edge_colors.append('#059669')  # Green for positive
                edge_styles.append('-')
            else:
                edge_colors.append('#dc2626')  # Red for negative
                edge_styles.append('--')
            edge_widths.append(min(abs(corr) * 8, 6))  # Cap maximum width
        
        nx.draw_networkx_edges(self.causal_graph, pos, 
                              width=edge_widths, alpha=0.7, 
                              edge_color=edge_colors, style=edge_styles)
        
        # Draw labels with better positioning
        nx.draw_networkx_labels(self.causal_graph, pos, node_labels, 
                               font_size=8, font_weight='bold', 
                               font_color='white', bbox=dict(boxstyle='round,pad=0.2', 
                               facecolor='black', alpha=0.7))
        
        plt.title('Enhanced Causal Graph: Feature Relationships', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#dc2626', 
                      markersize=12, label='Perturbation Flags'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3b82f6', 
                      markersize=10, label='Latent Features'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#f59e0b', 
                      markersize=10, label='Model Predictions'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#10b981', 
                      markersize=8, label='Physical Features'),
            plt.Line2D([0], [0], color='#059669', linewidth=3, label='Positive Correlation'),
            plt.Line2D([0], [0], color='#dc2626', linewidth=3, linestyle='--', label='Negative Correlation')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), 
                  frameon=True, fancybox=True, shadow=True)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'causal_graph_2d_enhanced.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        plt.close()
    
    def _create_3d_causal_graph(self, save_dir):
        """Create innovative 3D causal graph visualization."""
        if self.causal_graph.number_of_nodes() == 0:
            print("No nodes in causal graph for 3D visualization!")
            return
            
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Use 3D spring layout (approximated by adding z-coordinates)
        pos_2d = nx.spring_layout(self.causal_graph, k=1.5, iterations=200, seed=42)
        
        # Create 3D positions by adding z-coordinate based on node type
        pos_3d = {}
        node_colors = []
        node_sizes = []
        
        for node in self.causal_graph.nodes():
            x, y = pos_2d[node]
            
            # Assign z-coordinate based on node type for layered visualization
            if 'is_' in node and 'perturbed' in node:
                z = 2.0  # Top layer for perturbation flags
                node_colors.append('#dc2626')
                node_sizes.append(200)
            elif 'latent_' in node:
                z = 1.0  # Middle layer for latent features
                node_colors.append('#3b82f6')
                node_sizes.append(150)
            elif 'prob' in node:
                z = 1.5  # Between latent and perturbation
                node_colors.append('#f59e0b')
                node_sizes.append(180)
            elif 'system_' in node:
                z = 0.5  # Lower middle for system features
                node_colors.append('#8b5cf6')
                node_sizes.append(120)
            else:
                z = 0.0  # Bottom layer for physical features
                node_colors.append('#10b981')
                node_sizes.append(100)
                
            pos_3d[node] = (x, y, z)
        
        # Extract coordinates for plotting
        xs, ys, zs = zip(*pos_3d.values())
        
        # Plot nodes
        scatter = ax.scatter(xs, ys, zs, c=node_colors, s=node_sizes, 
                           alpha=0.8, edgecolors='white', linewidth=1, 
                           depthshade=True)
        
        # Plot edges with 3D lines
        for u, v, data in self.causal_graph.edges(data=True):
            x1, y1, z1 = pos_3d[u]
            x2, y2, z2 = pos_3d[v]
            
            corr = data['correlation']
            color = '#059669' if corr > 0 else '#dc2626'
            alpha = min(abs(corr) * 2, 1.0)  # Alpha based on correlation strength
            linewidth = abs(corr) * 4
            
            ax.plot([x1, x2], [y1, y2], [z1, z2], 
                   color=color, alpha=alpha, linewidth=linewidth)
        
        # Add node labels (selective to avoid clutter)
        important_nodes = [node for node in self.causal_graph.nodes() 
                          if 'latent_' in node or 'is_perturbed' in node or 'prob' in node]
        
        for node in important_nodes:
            x, y, z = pos_3d[node]
            label = node.replace('is_perturbed', 'Perturbed')\
                       .replace('perturbation_prob_model', 'PredProb')\
                       .replace('latent_', 'L')
            ax.text(x, y, z+0.1, label, fontsize=8, fontweight='bold')
        
        # Styling
        ax.set_xlabel('Network Dimension X', fontweight='bold', labelpad=10)
        ax.set_ylabel('Network Dimension Y', fontweight='bold', labelpad=10)
        ax.set_zlabel('Feature Layer', fontweight='bold', labelpad=10)
        ax.set_title('3D Causal Graph: Multi-Layer Feature Network', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Set z-axis labels
        ax.set_zticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.set_zticklabels(['Physical', 'System', 'Latent', 'Predictions', 'Perturbations'])
        
        # Enhanced grid and styling
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        ax.grid(True, alpha=0.3)
        
        # Optimal viewing angle
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'causal_graph_3d_enhanced.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        plt.close()
        
        print("Enhanced 3D causal graph visualization created!")
    
    def granger_causality_analysis(self, max_lag=5, p_threshold=0.05):
        """
        Perform Granger causality analysis on the full latent time series for each original simulation file.
        Returns a dictionary where keys are filenames and values are causality matrices.
        """
        if not STATSMODELS_AVAILABLE:
            print("Granger Causality: statsmodels not available. Skipping.")
            return None
        if not self.latent_analyzer.full_latent_sequences_per_file:
            print("No full latent sequences available for Granger Causality. Run extract_latent_representations first.")
            return None
            
        granger_causal_links = {}
        
        for file_path, z_seq_full_np in self.latent_analyzer.full_latent_sequences_per_file.items():
            file_name_key = os.path.basename(file_path).replace('.npz', '')
            
            if z_seq_full_np.shape[0] < max_lag + 2:
                print(f"Skipping {file_name_key} for Granger: not enough time points ({z_seq_full_np.shape[0]}) for max_lag={max_lag}. Requires at least {max_lag + 2}.")
                continue

            print(f"Performing Granger causality for {file_name_key} (full latent sequence)...")
            
            num_latent_dims = z_seq_full_np.shape[1]
            causality_matrix_p_values = np.ones((num_latent_dims, num_latent_dims))
            
            for i in range(num_latent_dims):
                for j in range(num_latent_dims):
                    if i != j:
                        try:
                            data_for_granger = np.column_stack([z_seq_full_np[:, j], z_seq_full_np[:, i]])
                            results = grangercausalitytests(data_for_granger, max_lag, verbose=False)
                            
                            p_value_at_max_lag = 1.0
                            if max_lag in results and 'ssr_ftest' in results[max_lag][0]:
                                p_value_at_max_lag = results[max_lag][0]['ssr_ftest'][1]
                            elif len(results) > 0 and 1 in results and 'ssr_ftest' in results[1][0]:
                                p_value_at_max_lag = results[1][0]['ssr_ftest'][1]
                            
                            causality_matrix_p_values[j, i] = p_value_at_max_lag
                                
                        except ValueError as ve:
                            print(f"  Granger test failed for {i}->{j} in {file_name_key} (ValueError): {ve}. Skipping.")
                        except Exception as e:
                            print(f"  Unexpected error in Granger for {i}->{j} in {file_name_key}: {e}. Skipping.")

            granger_causal_links[file_name_key] = causality_matrix_p_values
        
        return granger_causal_links
    
    def analyze_perturbation_effects(self, save_dir=os.path.join(RESULTS_DIR, 'causal_analysis')):
        """Analyze how different perturbations affect the features (using t-tests)."""
        os.makedirs(save_dir, exist_ok=True)

        if self.causal_data is None:
            print("No causal data available! Run prepare_causal_dataset first.")
            return

        clean_data = self.causal_data[self.causal_data['is_perturbed'] == 0]
        perturbed_data = self.causal_data[self.causal_data['is_perturbed'] == 1]

        if len(clean_data) == 0 or len(perturbed_data) == 0:
            print("Need both clean and perturbed data for comparison in analyze_perturbation_effects!")
            return

        feature_cols = [col for col in self.causal_data.columns 
                        if col.startswith(('latent_', 'body', 'system_', 'pair_'))]

        significant_differences = {}

        for feature in feature_cols:
            clean_values = clean_data[feature].values
            perturbed_values = perturbed_data[feature].values

            if len(clean_values) > 1 and len(perturbed_values) > 1 and \
               np.var(clean_values) > 1e-9 and np.var(perturbed_values) > 1e-9:
                try:
                    t_stat, p_value = stats.ttest_ind(clean_values, perturbed_values, equal_var=False)

                    if p_value < 0.05:
                        significant_differences[feature] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'clean_mean': np.mean(clean_values),
                            'perturbed_mean': np.mean(perturbed_values),
                            'effect_size': (np.mean(perturbed_values) - np.mean(clean_values)) / np.std(np.concatenate([clean_values, perturbed_values]))
                        }
                except Exception as e:
                    print(f"  T-test failed for feature {feature}: {e}. Skipping.")

        print("\n=== PERTURBATION EFFECT ANALYSIS (Statistical Differences) ===")
        print(f"Found {len(significant_differences)} features with significant differences (p < 0.05):")

        if not significant_differences:
            print("No significant differences found between clean and perturbed features.")
        else:
            for feature, stats_dict in sorted(significant_differences.items(), 
                                              key=lambda x: x[1]['p_value']):
                print(f"\n- {feature}:")
                print(f"  p-value: {stats_dict['p_value']:.6f}")
                print(f"  Effect size (Cohen's d): {stats_dict['effect_size']:.3f}")
                print(f"  Clean mean: {stats_dict['clean_mean']:.4f}")
                print(f"  Perturbed mean: {stats_dict['perturbed_mean']:.4f}")

        return significant_differences


def main():
    """Main causal discovery pipeline with enhanced 3D visualizations."""
    print("=== ENHANCED CAUSAL DISCOVERY ANALYSIS ===")
    
    # Define model and scaler filenames
    MODEL_FILE = 'anti_overfitting_best.pth'
    SCALER_FILE = 'robust_scaler.pkl'
    
    # Construct full paths to model and scaler
    model_path = os.path.join(MODELS_CHECKPOINT_DIR, MODEL_FILE)
    scaler_path = os.path.join(MODELS_CHECKPOINT_DIR, SCALER_FILE)

    # Check for trained model and scaler existence
    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}")
        print("Please train the model using overfitting_solution.py first!")
        return
    if not os.path.exists(scaler_path):
        print(f"Scaler not found at {scaler_path}")
        print("Please ensure the scaler was saved during training!")
        return
        
    # Initialize latent space analyzer with model and scaler
    analyzer = LatentSpaceAnalyzer(model_path, scaler_path)
    
    # Collect all data files
    all_simulation_files = []
    for root, _, files in os.walk(DATA_ROOT_DIR): 
        for filename in files:
            if filename.endswith('.npz'):
                all_simulation_files.append(os.path.join(root, filename))
                
    if not all_simulation_files:
        print(f"No data files found in {DATA_ROOT_DIR}! Please ensure simulator_fixed.py has generated data.")
        return
    
    # Extract latent representations for analysis
    print(f"\nExtracting latent representations from {len(all_simulation_files)} files...")
    analyzer.extract_latent_representations(all_simulation_files, sequence_length=10, prediction_horizon=2)
    
    if not analyzer.latent_representations_segments:
        print("No latent representations extracted. Check data files or model loading/processing.")
        return

    # Enhanced visualizations
    print("\nCreating enhanced latent space visualizations...")
    pca_result, latent_2d_plot_data = analyzer.visualize_latent_space()
    
    print("\nCreating 3D trajectory visualizations...")
    analyzer.create_interactive_3d_trajectory_visualization()
    
    # Build causal graph with enhanced 3D visualization
    print("\nBuilding enhanced causal graph...")
    graph_builder = CausalGraphBuilder(analyzer)
    causal_data_df = graph_builder.prepare_causal_dataset()
    
    if causal_data_df is not None and not causal_data_df.empty:
        # Build and visualize correlation-based graph (now includes both 2D and 3D)
        causal_graph_obj = graph_builder.build_correlation_graph(threshold=0.3)
        
        # Analyze statistical differences between clean and perturbed features
        print("\nAnalyzing perturbation effects on features...")
        significant_diffs_result = graph_builder.analyze_perturbation_effects()
        
        # Granger causality analysis
        print("\nPerforming Granger causality analysis...")
        granger_results_dict = graph_builder.granger_causality_analysis(max_lag=5, p_threshold=0.01)
        
        total_granger_links = 0
        if granger_results_dict:
            for mtx in granger_results_dict.values():
                total_granger_links += np.sum(mtx < 0.01)
            print(f"Granger causality analysis completed. Found {total_granger_links} significant causal links.")
        else:
            print("No Granger causality results available.")
            
        # Save summary results
        results_summary_file = os.path.join(RESULTS_DIR, 'enhanced_causal_analysis_summary.txt')
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        with open(results_summary_file, 'w') as f:
            f.write("ENHANCED THREE-BODY CAUSAL DISCOVERY RESULTS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total data segments analyzed: {len(analyzer.latent_representations_segments)}\n")
            f.write(f"Latent dimensions: {analyzer.model.latent_dim}\n")
            f.write(f"Causal graph nodes: {causal_graph_obj.number_of_nodes() if causal_graph_obj else 'N/A'}\n")
            f.write(f"Causal graph edges: {causal_graph_obj.number_of_edges() if causal_graph_obj else 'N/A'}\n")
            f.write(f"Significant feature differences: {len(significant_diffs_result) if significant_diffs_result else 0}\n")
            f.write(f"Granger causal links found: {total_granger_links}\n")
            f.write("\nEnhanced Visualizations Created:\n")
            f.write("- Enhanced 3D latent space plots with multiple views\n")
            f.write("- 3D trajectory evolution visualization\n")
            f.write("- Enhanced 2D causal graph with improved styling\n")
            f.write("- 3D multi-layer causal graph visualization\n")
            f.write("- Probability-colored 3D scatter plots\n")
            
            if significant_diffs_result:
                f.write("\nSignificant Feature Differences (p < 0.05):\n")
                for feature, stats_dict in sorted(significant_diffs_result.items(), key=lambda x: x[1]['p_value']):
                    f.write(f"- {feature}: p={stats_dict['p_value']:.6f}, Effect={stats_dict['effect_size']:.3f}\n")

        print(f"\nEnhanced analysis summary saved to: {results_summary_file}")
            
    print("\n=== ENHANCED CAUSAL DISCOVERY COMPLETE ===")
    print("Check the results directory for stunning 3D visualizations!")


if __name__ == '__main__':
    # Set enhanced plotting style for better visual appeal
    plt.style.use('default')  # Reset to default first
    sns.set_palette("husl")   # Use a vibrant color palette
    
    # Enhanced matplotlib settings for 3D plots
    plt.rcParams.update({
        'font.size': 11,
        'font.weight': 'normal',
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.titleweight': 'bold',
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'patch.linewidth': 0.5,
        'patch.facecolor': '348ABD',  # Nice blue
        'patch.edgecolor': 'eeeeee',
        'patch.antialiased': True,
        'axes.facecolor': 'white',
        'axes.edgecolor': 'bcbcbc',
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': 'b0b0b0',
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.3,
        'savefig.dpi': 300,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none'
    })

    main()