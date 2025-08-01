# three_body_causal_ai/notebooks/data_analyzer.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import os
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ThreeBodyAnalyzer:
    def __init__(self, data_dir='../data'):
        self.data_dir = data_dir
        self.clean_data = {}
        self.perturbed_data = {}
        
    def load_all_data(self):
        """Load all simulation data files."""
        clean_dir = os.path.join(self.data_dir, 'clean')
        perturbed_dir = os.path.join(self.data_dir, 'perturbed')
        
        # Load clean data
        if os.path.exists(clean_dir):
            for filename in os.listdir(clean_dir):
                if filename.endswith('.npz'):
                    key = filename.replace('.npz', '')
                    self.clean_data[key] = np.load(os.path.join(clean_dir, filename), allow_pickle=True)
                    print(f"Loaded clean data: {key}")
        
        # Load perturbed data  
        if os.path.exists(perturbed_dir):
            for filename in os.listdir(perturbed_dir):
                if filename.endswith('.npz'):
                    key = filename.replace('.npz', '')
                    self.perturbed_data[key] = np.load(os.path.join(perturbed_dir, filename), allow_pickle=True)
                    print(f"Loaded perturbed data: {key}")
    
    def extract_trajectories(self, data):
        """Extract individual body trajectories from solution array."""
        solution = data['solution']
        n_bodies = len(data['masses'])
        
        trajectories = []
        for i in range(n_bodies):
            # Each body has 6 components: x, y, z, vx, vy, vz
            pos_x = solution[:, i*6]
            pos_y = solution[:, i*6 + 1] 
            pos_z = solution[:, i*6 + 2]
            vel_x = solution[:, i*6 + 3]
            vel_y = solution[:, i*6 + 4]
            vel_z = solution[:, i*6 + 5]
            
            trajectories.append({
                'position': np.column_stack([pos_x, pos_y, pos_z]),
                'velocity': np.column_stack([vel_x, vel_y, vel_z])
            })
        
        return trajectories
    
    def plot_comparative_trajectories(self, clean_key, perturbed_key, save_fig=True):
        """Compare clean vs perturbed trajectories."""
        if clean_key not in self.clean_data or perturbed_key not in self.perturbed_data:
            print(f"Data not found: {clean_key} or {perturbed_key}")
            return
        
        clean_data = self.clean_data[clean_key]
        perturbed_data = self.perturbed_data[perturbed_key]
        
        clean_traj = self.extract_trajectories(clean_data)
        perturbed_traj = self.extract_trajectories(perturbed_data)
        
        fig = plt.figure(figsize=(15, 6))
        
        # Clean trajectories
        ax1 = fig.add_subplot(121, projection='3d')
        colors = ['red', 'blue', 'green']
        for i, traj in enumerate(clean_traj):
            ax1.plot(traj['position'][:, 0], 
                    traj['position'][:, 1], 
                    traj['position'][:, 2], 
                    color=colors[i], 
                    label=f'Body {i+1}', 
                    alpha=0.7,
                    linewidth=1)
            # Mark start and end
            ax1.scatter(*traj['position'][0], color=colors[i], s=50, marker='o')
            ax1.scatter(*traj['position'][-1], color=colors[i], s=50, marker='s')
        
        ax1.set_title(f'Clean: {clean_key}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y') 
        ax1.set_zlabel('Z')
        ax1.legend()
        
        # Perturbed trajectories
        ax2 = fig.add_subplot(122, projection='3d')
        for i, traj in enumerate(perturbed_traj):
            ax2.plot(traj['position'][:, 0], 
                    traj['position'][:, 1], 
                    traj['position'][:, 2], 
                    color=colors[i], 
                    label=f'Body {i+1}', 
                    alpha=0.7,
                    linewidth=1)
            ax2.scatter(*traj['position'][0], color=colors[i], s=50, marker='o')
            ax2.scatter(*traj['position'][-1], color=colors[i], s=50, marker='s')
        
        perturbation_type = perturbed_data['perturbation_type']
        ax2.set_title(f'Perturbed: {perturbation_type}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_fig:
            os.makedirs('../results/plots', exist_ok=True)
            plt.savefig(f'../results/plots/comparison_{clean_key}_vs_{perturbed_key}.png', 
                       dpi=300, bbox_inches='tight')
            print(f"Plot saved: comparison_{clean_key}_vs_{perturbed_key}.png")
        
        plt.show()
    
    def analyze_energy_conservation(self, data_key, is_clean=True):
        """Analyze energy conservation (should reveal perturbations)."""
        data_dict = self.clean_data if is_clean else self.perturbed_data
        
        if data_key not in data_dict:
            print(f"Data not found: {data_key}")
            return
        
        data = data_dict[data_key]
        trajectories = self.extract_trajectories(data)
        masses = data['masses']
        G = data['G_constant']
        time = data['time']
        
        total_energy = []
        kinetic_energy = []
        potential_energy = []
        
        for t_idx in range(len(time)):
            # Kinetic energy
            KE = 0
            for i, traj in enumerate(trajectories):
                v_squared = np.sum(traj['velocity'][t_idx]**2)
                KE += 0.5 * masses[i] * v_squared
            
            # Potential energy
            PE = 0
            for i in range(len(trajectories)):
                for j in range(i+1, len(trajectories)):
                    r_vec = trajectories[i]['position'][t_idx] - trajectories[j]['position'][t_idx]
                    r_mag = np.linalg.norm(r_vec)
                    if r_mag > 1e-10:
                        PE -= G * masses[i] * masses[j] / r_mag
            
            kinetic_energy.append(KE)
            potential_energy.append(PE)
            total_energy.append(KE + PE)
        
        # Plot energy evolution
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0,0].plot(time, kinetic_energy, 'b-', label='Kinetic')
        axes[0,0].set_title('Kinetic Energy')
        axes[0,0].set_ylabel('Energy')
        axes[0,0].grid(True)
        
        axes[0,1].plot(time, potential_energy, 'r-', label='Potential')
        axes[0,1].set_title('Potential Energy')
        axes[0,1].set_ylabel('Energy')
        axes[0,1].grid(True)
        
        axes[1,0].plot(time, total_energy, 'g-', label='Total')
        axes[1,0].set_title('Total Energy')
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Energy')
        axes[1,0].grid(True)
        
        # Energy variation (should be near zero for clean, nonzero for perturbed)
        energy_variation = np.array(total_energy) - total_energy[0]
        axes[1,1].plot(time, energy_variation, 'purple', label='Energy Drift')
        axes[1,1].set_title('Energy Conservation Check')
        axes[1,1].set_xlabel('Time')
        axes[1,1].set_ylabel('ΔE from initial')
        axes[1,1].grid(True)
        
        perturbation_info = "Clean" if is_clean else str(data['perturbation_type'])
        fig.suptitle(f'Energy Analysis: {data_key} ({perturbation_info})')
        plt.tight_layout()
        
        # Print statistics
        energy_drift_std = np.std(energy_variation)
        print(f"\nEnergy Analysis for {data_key}:")
        print(f"Energy drift std: {energy_drift_std:.2e}")
        print(f"Relative energy drift: {energy_drift_std/abs(total_energy[0]):.2e}")
        
        plt.show()
        return np.array(kinetic_energy), np.array(potential_energy), np.array(total_energy)
    
    def compute_trajectory_features(self, data_key, is_clean=True):
        """Extract features that could be used for ML training."""
        data_dict = self.clean_data if is_clean else self.perturbed_data
        
        if data_key not in data_dict:
            print(f"Data not found: {data_key}")
            return None
        
        data = data_dict[data_key]
        trajectories = self.extract_trajectories(data)
        time = data['time']
        
        features = {}
        
        for i, traj in enumerate(trajectories):
            body_features = {}
            
            # Position statistics
            pos = traj['position']
            body_features['pos_mean'] = np.mean(pos, axis=0)
            body_features['pos_std'] = np.std(pos, axis=0)
            body_features['pos_range'] = np.ptp(pos, axis=0)  # Peak-to-peak
            
            # Velocity statistics
            vel = traj['velocity']
            body_features['vel_mean'] = np.mean(vel, axis=0)
            body_features['vel_std'] = np.std(vel, axis=0)
            body_features['speed_max'] = np.max(np.linalg.norm(vel, axis=1))
            
            # Distance from origin
            distances = np.linalg.norm(pos, axis=1)
            body_features['dist_from_origin_mean'] = np.mean(distances)
            body_features['dist_from_origin_std'] = np.std(distances)
            
            # Acceleration (numerical derivative of velocity)
            if len(time) > 1:
                dt = time[1] - time[0]
                accel = np.gradient(vel, dt, axis=0)
                body_features['accel_mean'] = np.mean(np.linalg.norm(accel, axis=1))
                body_features['accel_std'] = np.std(np.linalg.norm(accel, axis=1))
            
            features[f'body_{i}'] = body_features
        
        # Pairwise features (distances between bodies)
        for i in range(len(trajectories)):
            for j in range(i+1, len(trajectories)):
                pairwise_distances = []
                for t_idx in range(len(time)):
                    dist = np.linalg.norm(
                        trajectories[i]['position'][t_idx] - trajectories[j]['position'][t_idx]
                    )
                    pairwise_distances.append(dist)
                
                features[f'pair_{i}_{j}'] = {
                    'distance_mean': np.mean(pairwise_distances),
                    'distance_std': np.std(pairwise_distances),
                    'distance_min': np.min(pairwise_distances),
                    'distance_max': np.max(pairwise_distances)
                }
        
        return features
    
    def generate_dataset_summary(self):
        """Generate a comprehensive summary of all loaded data."""
        print("=== DATASET SUMMARY ===")
        print(f"Clean simulations: {len(self.clean_data)}")
        print(f"Perturbed simulations: {len(self.perturbed_data)}")
        
        # Analyze all datasets
        all_features = {}
        
        for key, data in self.clean_data.items():
            features = self.compute_trajectory_features(key, is_clean=True)
            if features:
                all_features[key] = {'features': features, 'label': 'clean'}
        
        for key, data in self.perturbed_data.items():
            features = self.compute_trajectory_features(key, is_clean=False)
            if features:
                all_features[key] = {
                    'features': features, 
                    'label': 'perturbed',
                    'perturbation_type': str(data['perturbation_type'])
                }
        
        # Save feature summary
        summary_file = '../results/dataset_features_summary.txt'
        os.makedirs('../results', exist_ok=True)
        
        with open(summary_file, 'w') as f:
            f.write("THREE-BODY CAUSAL AI - DATASET SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            for key, info in all_features.items():
                f.write(f"Dataset: {key}\n")
                f.write(f"Type: {info['label']}\n")
                if 'perturbation_type' in info:
                    f.write(f"Perturbation: {info['perturbation_type']}\n")
                f.write("-" * 30 + "\n")
                
                # Write key statistics
                features = info['features']
                for body_key, body_features in features.items():
                    if body_key.startswith('body_'):
                        f.write(f"  {body_key}:\n")
                        f.write(f"    Max speed: {body_features.get('speed_max', 'N/A'):.4f}\n")
                        f.write(f"    Avg distance from origin: {body_features.get('dist_from_origin_mean', 'N/A'):.4f}\n")
                f.write("\n")
        
        print(f"Dataset summary saved to: {summary_file}")
        return all_features

def main():
    """Main analysis workflow."""
    analyzer = ThreeBodyAnalyzer()
    
    print("Loading all simulation data...")
    analyzer.load_all_data()
    
    if not analyzer.clean_data and not analyzer.perturbed_data:
        print("No data found! Run the simulator first.")
        return
    
    # Generate comprehensive summary
    print("\nGenerating dataset summary...")
    features = analyzer.generate_dataset_summary()
    
    # Compare trajectories if we have both clean and perturbed data
    if analyzer.clean_data and analyzer.perturbed_data:
        clean_key = list(analyzer.clean_data.keys())[0]
        perturbed_key = list(analyzer.perturbed_data.keys())[0]
        
        print(f"\nComparing trajectories: {clean_key} vs {perturbed_key}")
        analyzer.plot_comparative_trajectories(clean_key, perturbed_key)
        
        print(f"\nAnalyzing energy conservation for clean data...")
        analyzer.analyze_energy_conservation(clean_key, is_clean=True)
        
        print(f"\nAnalyzing energy conservation for perturbed data...")
        analyzer.analyze_energy_conservation(perturbed_key, is_clean=False)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Ready for next phase: AI model training!")

if __name__ == '__main__':
    main()