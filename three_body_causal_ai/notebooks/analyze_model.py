# three_body_causal_ai/notebooks/analyze_model.py

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle # For loading the scaler
from sklearn.preprocessing import RobustScaler # Ensure this is imported for scaler type
from torch.utils.data import DataLoader, default_collate # Needed for loading data for inference, and default_collate
from sklearn.decomposition import PCA # For latent space PCA analysis
import seaborn as sns # For plotting aesthetics
from sklearn import metrics # <--- ADD THIS IMPORT at the top of the file
# --- Import necessary classes from your overfitting_solution.py ---
import sys
# Adjust path: 'notebooks' is inside 'three_body_causal_ai', 'model' is also inside 'three_body_causal_ai'
# So, to get to 'model' from 'notebooks', you go up one ('..') then down into 'model'.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../model'))
from overfitting_solution import UltraMinimalNeuralODE, ImprovedDataset # Import your model and dataset classes

# --- Path setup (consistent with overfitting_solution.py) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT should go up one level from 'notebooks/' to 'three_body_causal_ai/'
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..') 
DATA_ROOT_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'model', 'checkpoints')
SCALER_PATH = os.path.join(MODELS_CHECKPOINT_DIR, 'robust_scaler.pkl') 

# Ensure seaborn style is set up (optional)
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis") # A nice colormap

print("=== ANALYZE TRAINED MODEL ===")

# --- Custom Collate Function (copied from overfitting_solution.py to avoid re-importing main script) ---
# This is crucial for correctly batching sequences, targets, and metadata (dictionaries)
def custom_collate_fn(batch):
    """Custom collate function for DataLoader."""
    sequences = [item[0] for item in batch]
    targets = [item[1] for item in batch] # CRITICAL FIX: Changed from item[0] to item[1]
    metadata_list = [item[2] for item in batch]
    
    collated_sequences = default_collate(sequences)
    collated_targets = default_collate(targets)
    
    return collated_sequences, collated_targets, metadata_list

# --- Step 1: Load the Scaler ---
print(f"Loading scaler from: {SCALER_PATH}")
try:
    with open(SCALER_PATH, 'rb') as f:
        loaded_scaler = pickle.load(f)
    print("Scaler loaded successfully.")
except FileNotFoundError:
    print(f"Error: Scaler file not found at {SCALER_PATH}. Please ensure it was saved during training (in overfitting_solution.py).")
    exit()
except Exception as e:
    print(f"Error loading scaler: {e}")
    exit()

# --- Step 2: Load the Model ---
MODEL_FILENAME = 'anti_overfitting_best.pth' # Ensure this matches the file saved by AntiOverfittingTrainer
model_full_path = os.path.join(MODELS_CHECKPOINT_DIR, MODEL_FILENAME)

print(f"Loading model from: {model_full_path}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Instantiate the model with the exact architecture parameters it was trained with
# (These MUST match UltraMinimalNeuralODE.__init__ in overfitting_solution.py)
model_params = {
    'state_dim': 18,
    'latent_dim': 3,  # CORRECTED: This must match UltraMinimalNeuralODE's __init__
    'hidden_dim': 12, # CORRECTED: This must match UltraMinimalNeuralODE's __init__
    'dropout_rate': 0.5 # CORRECTED: This must match UltraMinimalNeuralODE's __init__
}
loaded_model = UltraMinimalNeuralODE(**model_params).to(device)

# Load the saved state dictionary
try:
    checkpoint = torch.load(model_full_path, map_location=device)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval() # Set model to evaluation mode (important for dropout, batchnorm behavior)
    print("Model loaded successfully and set to eval mode.")
    print(f"Model was saved at Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}, Ratio: {checkpoint['val_ratio']:.2f}")
except FileNotFoundError:
    print(f"Error: Model checkpoint not found at {model_full_path}. Please ensure training was successful (ran overfitting_solution.py).")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# --- Step 3: Prepare Data for Inference/Analysis ---
# This will load segments from existing .npz files for analysis
all_files_for_inference = []
for root, _, files in os.walk(DATA_ROOT_DIR):
    for filename in files:
        if filename.endswith('.npz'):
            all_files_for_inference.append(os.path.join(root, filename))

if not all_files_for_inference:
    print(f"Error: No .npz data files found in {DATA_ROOT_DIR}. Please run simulator_fixed.py first.")
    exit()

# Create a dataset for inference, using the same segmentation parameters as training
# (sequence_length=10, prediction_horizon=2 from UltraMinimalNeuralODE training in overfitting_solution.py)
inference_dataset = ImprovedDataset(
    all_files_for_inference, # Use all files for more diverse analysis samples
    scaler=loaded_scaler,    # Crucially, pass the *loaded* scaler
    fit_scaler=False,        # Do NOT fit scaler here
    augment_data=False,      # Do NOT augment analysis data
    balance_classes=False,    # Do NOT balance classes here
    sequence_length=10,      # Must match training sequence_length
    prediction_horizon=2     # Must match training prediction_horizon
)
# Use a small batch size for individual sample analysis or a larger one for overall metrics
inference_loader = DataLoader(inference_dataset, batch_size=min(16, len(inference_dataset)), shuffle=True, collate_fn=custom_collate_fn) 

print(f"\nReady for inference and analysis with {len(inference_dataset)} samples.")

# --- Step 4: Perform Inference and Basic Visualization ---
num_samples_to_plot = 5 # Plot first 5 samples for visual inspection

# Collect latent means for broader visualization later
all_latent_means = []
all_true_is_perturbed = []
all_predicted_prob = []
all_perturbation_types = []

print("\nPerforming inference and plotting sample trajectories...")
for i, (sequence, target, metadata) in enumerate(inference_loader):
    # Process all batches to collect latent means, but only plot a few
    
    sequence = sequence.to(device)
    target = target.to(device) # True future states

    with torch.no_grad():
        t_span = torch.arange(target.size(1)).float().to(device)
        # Unpack only what UltraMinimalNeuralODE.forward returns
        x_recon, x_pred, perturbation_prob, z_seq = loaded_model(sequence, t_span)

    # Move to CPU and convert to numpy for plotting/collection
    # Squeeze is used to remove singleton dimensions (e.g., batch_size=1)
    sequence_np = sequence.cpu().numpy().squeeze() 
    target_np = target.cpu().numpy().squeeze()
    x_recon_np = x_recon.cpu().numpy().squeeze()
    x_pred_np = x_pred.cpu().numpy().squeeze()
    z_seq_np = z_seq.cpu().numpy().squeeze()
    
    # Handle squeeze output shape for batch_size > 1 or single item in batch
    # Ensures current_xyz_np will always be (sequence_length, state_dim) for plotting one sample
    if sequence_np.ndim == 3: # Means batch_size > 1, so take the first item from the batch
        current_seq_np = sequence_np[0]
        current_target_np = target_np[0]
        current_x_recon_np = x_recon_np[0]
        current_x_pred_np = x_pred_np[0]
        current_z_seq_np = z_seq_np[0]
        current_metadata_item = metadata[0]
        current_perturbation_prob = perturbation_prob[0].item()
    else: # Means batch_size was 1 (ndim=2) or less than 16, so squeeze removed batch dim entirely
        current_seq_np = sequence_np
        current_target_np = target_np
        current_x_recon_np = x_recon_np
        current_x_pred_np = x_pred_np
        current_z_seq_np = z_seq_np
        current_metadata_item = metadata[0] # metadata is always a list of dicts
        current_perturbation_prob = perturbation_prob.item() # If batch_size=1, perturbation_prob is scalar
        

    # UN-NORMALIZE the data for physical interpretation before plotting!
    current_seq_unnorm = loaded_scaler.inverse_transform(current_seq_np)
    current_target_unnorm = loaded_scaler.inverse_transform(current_target_np)
    current_x_recon_unnorm = loaded_scaler.inverse_transform(current_x_recon_np)
    current_x_pred_unnorm = loaded_scaler.inverse_transform(current_x_pred_np)

    # Store latent means for later comprehensive PCA plotting
    # Loop through all items in the batch to collect their latent means
    for b_item in range(sequence.shape[0]): # sequence.shape[0] is the actual batch size
        all_latent_means.append(torch.mean(z_seq[b_item], dim=0).cpu().numpy())
        all_true_is_perturbed.append(metadata[b_item]['is_perturbed'])
        all_predicted_prob.append(perturbation_prob[b_item].item())
        all_perturbation_types.append(metadata[b_item]['perturbation_type'])


    # Plot only a few samples for visual inspection
    if i < num_samples_to_plot: # Only plot for the first 'num_samples_to_plot' batches
        print(f"\nPlotting Batch {i+1} Item 1 from file: {current_metadata_item['file']}")
        print(f"Perturbation Type: {current_metadata_item['perturbation_type']}, True Perturbed: {current_metadata_item['is_perturbed']}")
        print(f"Model Predicted Perturbation Probability: {current_perturbation_prob:.4f}")

        # --- Plot Trajectories (Body 1) ---
        fig_traj = plt.figure(figsize=(10, 8)) # Create ONE figure object
        ax = fig_traj.add_subplot(111, projection='3d') # Add subplot to THAT figure object

        # Plotting for Body 1 (indices 0,1,2 for position)
        ax.plot(current_seq_unnorm[:, 0], current_seq_unnorm[:, 1], current_seq_unnorm[:, 2], 'b-', label='Original Input (Body 1)', alpha=0.7)
        ax.plot(current_x_recon_unnorm[:, 0], current_x_recon_unnorm[:, 1], current_x_recon_unnorm[:, 2], 'b--', label='Reconstructed Input (Body 1)', alpha=0.9, linestyle=':')
        
        ax.plot(current_target_unnorm[:, 0], current_target_unnorm[:, 1], current_target_unnorm[:, 2], 'g-', label='True Future (Body 1)', alpha=0.7)
        ax.plot(current_x_pred_unnorm[:, 0], current_x_pred_unnorm[:, 1], current_x_pred_unnorm[:, 2], 'r:', label='Predicted Future (Body 1)', alpha=0.9)

        ax.set_title(f"Body 1 Trajectory ({current_metadata_item['perturbation_type']} - P_pred:{current_perturbation_prob:.2f})")
        ax.set_xlabel('X Position'); ax.set_ylabel('Y Position'); ax.set_zlabel('Z Position')
        ax.legend()
        ax.grid(True)
        plt.show(block=False) # Display plot without blocking
        plt.pause(0.1) # Short pause to allow rendering
        plt.close(fig_traj) # Close the specific figure


        # --- Plot Latent Space Trajectory for this single sample ---
        fig_latent_traj = plt.figure(figsize=(8, 6)) # Create ONE figure object
        if model_params['latent_dim'] == 3: 
            ax_latent = fig_latent_traj.add_subplot(111, projection='3d') # 3D plot for latent_dim=3
            ax_latent.plot(current_z_seq_np[:, 0], current_z_seq_np[:, 1], current_z_seq_np[:, 2], 'go-', markersize=3, label='Latent Z Trajectory')
            ax_latent.set_xlabel('Z0'); ax_latent.set_ylabel('Z1'); ax_latent.set_zlabel('Z2')
        else: # Fallback for 2D or other latent dims
            ax_latent = fig_latent_traj.add_subplot(111) # 2D plot
            ax_latent.plot(current_z_seq_np[:, 0], current_z_seq_np[:, 1], 'go-', markersize=3, label='Latent Z (Dim 0 vs Dim 1)')
            ax_latent.set_xlabel('Z0'); ax_latent.set_ylabel('Z1')
        
        ax_latent.set_title(f"Latent Space Trajectory for Sample {i+1} (Batch Item 1)")
        ax_latent.legend()
        ax_latent.grid(True)
        plt.show(block=False) # Display plot without blocking
        plt.pause(0.1)
        plt.close(fig_latent_traj) # Close the specific figure


# --- Step 5: Comprehensive Latent Space Analysis (PCA) ---
print("\n=== Comprehensive Latent Space Analysis ===")
if len(all_latent_means) == 0:
    print("No latent means collected for comprehensive analysis.")
else:
    all_latent_means_np = np.array(all_latent_means)
    all_true_is_perturbed_np = np.array(all_true_is_perturbed)
    all_predicted_prob_np = np.array(all_predicted_prob)
    all_perturbation_types_np = np.array(all_perturbation_types)

    # Perform PCA for 2D visualization if latent_dim > 2
    if all_latent_means_np.shape[1] > 2: # If latent dim > 2, do PCA
        pca = PCA(n_components=2)
        latent_2d_pca = pca.fit_transform(all_latent_means_np)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    else: # If latent dim is 1 or 2, just use it directly
        pca = None
        latent_2d_pca = all_latent_means_np[:, :min(all_latent_means_np.shape[1], 2)] # Take first 1 or 2 dims
        if latent_2d_pca.shape[1] == 1: # If only 1 dim, add a zero dim for 2D plot
            latent_2d_pca = np.column_stack([latent_2d_pca, np.zeros(latent_2d_pca.shape[0])])


    fig_pca, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Clean vs Perturbed (True Label)
    colors_binary = ['blue', 'red'] # Blue for clean, red for perturbed
    labels_binary = ['Clean', 'Perturbed']
    for i, label in enumerate(labels_binary):
        mask = all_true_is_perturbed_np == i
        if np.any(mask):
            axes[0].scatter(latent_2d_pca[mask, 0], latent_2d_pca[mask, 1], 
                            c=colors_binary[i], label=label, alpha=0.7, s=15)
    axes[0].set_title('Latent Space: True Clean vs Perturbed')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)' if pca else 'Latent Dim 0')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)' if pca else 'Latent Dim 1')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: By Specific Perturbation Type
    unique_types = sorted(list(set(all_perturbation_types_np)))
    cmap = plt.cm.get_cmap('tab10', len(unique_types))
    for i, ptype in enumerate(unique_types):
        mask = all_perturbation_types_np == ptype
        if np.any(mask):
            axes[1].scatter(latent_2d_pca[mask, 0], latent_2d_pca[mask, 1], 
                            c=[cmap(i)], label=ptype, alpha=0.7, s=15)
    axes[1].set_title('Latent Space: By Specific Perturbation Type')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)' if pca else 'Latent Dim 0')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)' if pca else 'Latent Dim 1')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(MODELS_CHECKPOINT_DIR, 'latent_space_overview.png'), dpi=300, bbox_inches='tight')
    plt.show() # Final show for PCA plots, usually left open for inspection
print("\n=== Perturbation Classification Performance ===")
if len(all_predicted_prob) == 0:
    print("No predictions collected for classification analysis.")
else:
    # Convert lists to NumPy arrays
    true_labels = np.array(all_true_is_perturbed)
    predicted_probs = np.array(all_predicted_prob)

    # Define a classification threshold (e.g., 0.5)
    # If probability >= 0.5, predict 1 (perturbed), else 0 (clean)
    predicted_labels = (predicted_probs >= 0.5).astype(int)

    # Calculate classification metrics
    accuracy = metrics.accuracy_score(true_labels, predicted_labels)
    precision = metrics.precision_score(true_labels, predicted_labels, zero_division=0) # zero_division handles no positive predictions
    recall = metrics.recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = metrics.f1_score(true_labels, predicted_labels, zero_division=0)
    
    # Generate confusion matrix
    conf_matrix = metrics.confusion_matrix(true_labels, predicted_labels)
    
    print(f"Classification Threshold: 0.5")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("  [[True Negatives, False Positives]")
    print("   [False Negatives, True Positives]]")

    # Optional: Plot Confusion Matrix for better visualization
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Clean', 'Predicted Perturbed'],
                yticklabels=['True Clean', 'True Perturbed'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_CHECKPOINT_DIR, 'confusion_matrix.png'), dpi=300)
    plt.show() # Display confusion matrix

print("\nAnalysis complete! Examine plots for insights into model performance and latent space structure.")
