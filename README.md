# Causal Discovery of Hidden Dynamics in Chaotic Three-Body Systems Using AI-Enhanced Latent Space Analysis

## Project Overview
This project explores the fascinating intersection of Artificial Intelligence and fundamental physics. It presents a novel AI-enhanced framework designed to **discover unobserved, 'hidden' forces** influencing chaotic dynamical systems, specifically focusing on the challenging three-body problem. By learning abstract, discriminative latent representations, our AI moves beyond mere prediction to provide concrete, data-driven hypotheses about the nature of anomalous physical phenomena.

## Key Contributions
* **Novel AI Framework:** Implements a Neural ODE Variational Autoencoder (NODE-VAE) tailored for causal discovery in complex dynamics.
* **Robust Data Pipeline:** Generates a large, diverse dataset of three-body interactions (clean and perturbed by 5 hidden force types) with advanced simulation stability techniques.
* **Overfitting Mitigation:** Incorporates sophisticated preprocessing and training strategies (e.g., RobustScaler, data augmentation, balanced classes, adaptive loss weighting, gradient clipping) to ensure strong model generalization.
* **Quantifiable Discovery:** Demonstrates the AI's ability to reliably detect hidden forces with high classification performance.
* **Interpretable Latent Space:** Shows statistical significance and visual separability of AI-discovered latent dimensions, linking them to physical observables via correlation analysis.

## Project Objectives
1.  Simulate diverse three-body systems with known (but hidden) perturbations.
2.  Develop an AI model capable of learning system dynamics and inferring latent representations.
3.  Apply causal discovery techniques to identify relationships between latent features, perturbations, and physical observables.
4.  Validate the AI's ability to detect and characterize hidden physics.

## Methodology
Our pipeline comprises:
1.  **High-Fidelity Simulation:** Custom Python environment using `scipy.integrate.solve_ivp` with adaptive solvers and softening, generating ~180 unique simulation files.
2.  **Data Preprocessing:** Segmentation of trajectories, `RobustScaler` normalization, class balancing, and strong data augmentation.
3.  **Ultra-Minimal Neural ODE VAE:** A compact model architecture (`latent_dim=3`) with encoder-decoder, ODE function, and a perturbation classifier head.
4.  **Robust Training:** `AdamW` optimizer, `ReduceLROnPlateau` scheduler, strong regularization, and balanced composite loss function.
5.  **Causal Analysis:** Statistical t-tests for feature differences, PCA for latent space visualization, and correlation-based graph construction using `networkx`.

## Performance Highlights
* **Model Generalization:** Achieved a best validation/training loss ratio of **0.57**, indicating excellent generalization.
* **Hidden Force Detection (Classification on Validation Set):**
    * Accuracy: **70.66%**
    * Precision: **84.30%**
    * Recall: **79.49%**
    * F1-Score: **0.8182**
* **Latent Space Discovery:** All three learned latent dimensions (`latent_0`, `latent_1`, `latent_2`) exhibited **statistically significant differences (p < 0.05)** between clean and perturbed systems.

## Project Structure
Three-Body-Causal-AI/
├── .git/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── three_body_causal_ai/
├── causal/
│   └── causal_discovery.py
├── data/
│   ├── clean/
│   └── perturbed/
├── model/
│   ├── checkpoints/
│   │   ├── anti_overfitting_best.pth
│   │   └── robust_scaler.pkl
│   └── overfitting_solution.py
├── notebooks/
│   ├── analyze_model.py
│   └── data_analyzer.py
├── results/
│   ├── causal_analysis/
│   ├── latent_analysis/
│   └── (analysis summaries, plots)
└── simulation/
└── simulator_fixed.py


## Setup and Usage
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Three-Body-Causal-AI.git](https://github.com/your-username/Three-Body-Causal-AI.git)
    cd Three-Body-Causal-AI
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: PyTorch (`torch`) installation might require specific commands based on your CUDA version. Refer to [PyTorch's website](https://pytorch.org/get-started/locally/) for details.*

4.  **Generate Data (This step takes significant time: ~8-12 hours):**
    ```bash
    cd three_body_causal_ai/simulation
    python simulator_fixed.py
    ```
    Ensure this completes and populates `three_body_causal_ai/data/` with hundreds of `.npz` files.

5.  **Analyze Data Quality (Optional but recommended):**
    ```bash
    cd three_body_causal_ai/notebooks
    python data_analyzer.py
    ```

6.  **Train the AI Model (This step takes significant time: ~1-3 hours):**
    ```bash
    cd three_body_causal_ai/model
    python overfitting_solution.py
    ```
    This will save the trained model (`anti_overfitting_best.pth`) and scaler (`robust_scaler.pkl`) in `three_body_causal_ai/model/checkpoints/`.

7.  **Analyze Model Performance and Latent Space:**
    ```bash
    cd three_body_causal_ai/notebooks
    python analyze_model.py
    ```
    This generates visual plots (saved in `three_body_causal_ai/model/checkpoints/` and displayed).

8.  **Perform Causal Discovery Analysis:**
    ```bash
    cd three_body_causal_ai/causal
    python causal_discovery.py
    ```
    This generates causal graph plots (saved in `three_body_causal_ai/results/causal_analysis/` and displayed) and a summary (`causal_analysis_summary.txt`).

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.


