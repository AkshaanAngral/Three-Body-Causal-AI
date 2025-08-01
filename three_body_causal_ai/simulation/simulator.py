# three_body_causal_ai/simulation/simulator_fixed.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import time
import warnings
import random
import logging

# Suppress integration warnings that are not critical errors (e.g., max_step warning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Path Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# --- Global Constants ---
G_CONST = 1.0 # Gravitational constant in dimensionless units
SOFTENING_PARAM = 0.001 # Further reduced softening for stronger close-range gravity.
# Adjust carefully: too small -> instability, too large -> non-physical force.

# --- Helper Functions ---
def get_initial_conditions(positions, velocities):
    """Convert positions and velocities to flat state vector for solve_ivp."""
    y0 = []
    for i in range(len(positions)):
        y0.extend(positions[i])
        y0.extend(velocities[i])
    return np.array(y0)

def n_body_ode_stable(t, y, masses, G, perturbation_type=None, perturbation_params=None):
    """
    N-body ODE with a standard softened gravitational potential.
    """
    n_bodies = len(masses)
    positions = y[:n_bodies*3].reshape(n_bodies, 3)
    velocities = y[n_bodies*3:].reshape(n_bodies, 3)
    
    dydt = np.zeros_like(y)
    accelerations = np.zeros_like(positions)
    
    # Newtonian gravity with standard softened potential
    for i in range(n_bodies):
        dydt[i*3 : i*3 + 3] = velocities[i] # dx/dt = v
        
        for j in range(n_bodies):
            if i != j:
                r_vec = positions[j] - positions[i]
                r_mag_sq = np.sum(r_vec**2)
                
                # Standard softened potential: (r^2 + epsilon^2)^(3/2) in denominator
                # Accel = (G * M * r_vec) / (r_mag_sq + SOFTENING_PARAM**2)**1.5
                denom = (r_mag_sq + SOFTENING_PARAM**2)**1.5
                
                # Avoid division by zero if denom is extremely small (shouldn't happen with softening unless epsilon=0)
                if denom < 1e-20: 
                    continue # Skip this pair or add a stronger safeguard
                
                accel_mag_dir = (G * masses[j]) / denom # Magnitude and direction combined
                accelerations[i] += accel_mag_dir * r_vec # Add to acceleration for body i
                
    # Apply perturbations (ensure consistency with softening/direction)
    if perturbation_type == 'hidden_mass' and perturbation_params:
        hm_pos = np.array(perturbation_params['position'])
        hm_mass = perturbation_params['mass']
        
        for i in range(n_bodies):
            r_vec_hm = hm_pos - positions[i]
            r_mag_hm_sq = np.sum(r_vec_hm**2)
            denom_hm = (r_mag_hm_sq + SOFTENING_PARAM**2)**1.5
            if denom_hm < 1e-20: continue
            
            accel_mag_dir_hm = (G * hm_mass) / denom_hm
            accelerations[i] += accel_mag_dir_hm * r_vec_hm
            
    elif perturbation_type == 'non_inverse_square' and perturbation_params:
        # Adds a 1/r^4 perturbation (still with softening)
        C = perturbation_params.get('constant', 1e-7) # Even smaller default for 1/r^4
        
        for i in range(n_bodies):
            for j in range(n_bodies):
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r_mag_sq = np.sum(r_vec**2)
                    
                    # Force proportional to 1/r^4, using softened distance
                    denom_nis = (r_mag_sq + SOFTENING_PARAM**2)**2 # (r^2 + eps^2)^2 for 1/r^4 behavior
                    if denom_nis < 1e-20: continue
                    
                    accel_nis_mag_dir = (C * masses[j]) / denom_nis 
                    accelerations[i] += accel_nis_mag_dir * r_vec
            
    elif perturbation_type == 'drag' and perturbation_params:
        drag_coeff = perturbation_params['coefficient']
        drag_body_idx = perturbation_params.get('body_index', None)
        
        for i in range(n_bodies):
            if drag_body_idx is None or i == drag_body_idx:
                v_mag_sq = np.sum(velocities[i]**2)
                if v_mag_sq > 1e-20: # Check for non-zero velocity
                    # Linear drag: F = -k * v, so a = -k * v / m
                    drag_accel = -drag_coeff * velocities[i] / masses[i]
                    
                    if 'zone_center' in perturbation_params and 'zone_radius' in perturbation_params:
                        zone_center = np.array(perturbation_params['zone_center'])
                        zone_radius = perturbation_params['zone_radius']
                        if np.linalg.norm(positions[i] - zone_center) < zone_radius:
                            accelerations[i] += drag_accel
                    else:
                        accelerations[i] += drag_accel
            
    elif perturbation_type == 'impulse' and perturbation_params:
        impulse_time_start = perturbation_params['time']
        impulse_duration = perturbation_params.get('duration', 0.1)
        impulse_body_idx = perturbation_params['body_index']
        impulse_vec = np.array(perturbation_params['vector'])
        
        if (impulse_body_idx < n_bodies and 
            t >= impulse_time_start and 
            t < impulse_time_start + impulse_duration):
            accelerations[impulse_body_idx] += (impulse_vec / impulse_duration) / masses[impulse_body_idx]
            
    elif perturbation_type == 'time_varying_G' and perturbation_params:
        alpha = perturbation_params.get('alpha', 0.005) 
        omega = perturbation_params.get('omega', 2 * np.pi / 20)
        
        current_G = G * (1 + alpha * np.sin(omega * t))
        
        accelerations_tvg = np.zeros_like(positions)
        for i in range(n_bodies):
            for j in range(n_bodies):
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r_mag_sq = np.sum(r_vec**2)
                    denom_tvg = (r_mag_sq + SOFTENING_PARAM**2)**1.5
                    if denom_tvg < 1e-20: continue
                    
                    accel_mag_dir_tvg = (current_G * masses[j]) / denom_tvg
                    accelerations_tvg[i] += accel_mag_dir_tvg * r_vec
        accelerations = accelerations_tvg
            
    # Assign accelerations to dydt (velocity derivatives)
    for i in range(n_bodies):
        dydt[n_bodies*3 + i*3 : n_bodies*3 + i*3 + 3] = accelerations[i]
        
    return dydt

def simulate_system_robust(
    masses,
    initial_positions,
    initial_velocities,
    time_span,
    num_steps,
    G_constant=G_CONST,
    perturbation_type=None,
    perturbation_params=None,
    output_dir=DATA_DIR,
    filename_prefix='simulation',
    save_data=True
):
    """
    Robust simulation with adaptive error handling (trying different ODE solvers and tolerances).
    """
    initial_conditions_flat = get_initial_conditions(initial_positions, initial_velocities)
    t_eval = np.linspace(time_span[0], time_span[1], num_steps)
    
    logger.info(f"Starting simulation: {filename_prefix}")
    logger.info(f"  Duration: {time_span[1] - time_span[0]:.2f} time units")
    logger.info(f"  Number of steps: {num_steps}")
    if perturbation_type:
        logger.info(f"  Perturbation: {perturbation_type} with params: {perturbation_params}")
    else:
        logger.info("  Clean Newtonian simulation.")
        
    start_time = time.time()
    
    # Ordered list of methods and tolerance pairs to try (tightest to loosest)
    attempts = [
        {'method': 'DOP853', 'rtol': 1e-10, 'atol': 1e-12}, # Very high accuracy
        {'method': 'DOP853', 'rtol': 1e-9, 'atol': 1e-11}, 
        {'method': 'RK45',   'rtol': 1e-8, 'atol': 1e-10},
        {'method': 'RK45',   'rtol': 1e-7, 'atol': 1e-9},
        {'method': 'LSODA',  'rtol': 1e-6, 'atol': 1e-8}, # More robust for stiff, less precise
        {'method': 'LSODA',  'rtol': 1e-5, 'atol': 1e-7},
    ]
    
    sol = None
    success_method = None
    
    for attempt_params in attempts:
        try:
            sol = solve_ivp(
                n_body_ode_stable,
                time_span,
                initial_conditions_flat,
                t_eval=t_eval,
                method=attempt_params['method'],
                args=(masses, G_constant, perturbation_type, perturbation_params),
                rtol=attempt_params['rtol'],
                atol=attempt_params['atol'],
                max_step=0.01 # Even smaller max_step for very chaotic/interacting systems
            )
            if sol.success:
                success_method = attempt_params['method']
                logger.info(f"  Successful with method={success_method}, rtol={attempt_params['rtol']}, atol={attempt_params['atol']}")
                break # Exit loop on first success
        except Exception as e:
            logger.warning(f"  Attempt with method={attempt_params['method']}, rtol={attempt_params['rtol']}, atol={attempt_params['atol']} failed: {e}")
            sol = None # Ensure sol is reset if it partially succeeded before full failure
            continue # Try next attempt
            
    end_time = time.time()
    logger.info(f"  Simulation completed in {end_time - start_time:.2f} seconds.")
    
    if not sol or not sol.success:
        logger.error(f"  ALL INTEGRATION ATTEMPTS FAILED for {filename_prefix}. Data will NOT be saved. Last message: {sol.message if sol else 'Solver failed to initialize.'}")
        return None, None
        
    time_points = sol.t
    solution_array = sol.y.T # Transpose to get (num_steps, num_variables)
    
    if save_data:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{filename_prefix}.npz")
        np.savez_compressed(filename, # Use compression to save space
             time=time_points, 
             solution=solution_array,
             masses=masses, 
             initial_positions=initial_positions,
             initial_velocities=initial_velocities,
             G_constant=G_constant,
             perturbation_type=perturbation_type, 
             perturbation_params=perturbation_params,
             solver_method=success_method, # Save which method succeeded
             description=f"Three-body simulation data ({filename_prefix})"
        )
        logger.info(f"  Data saved to {filename}")
        
    return time_points, solution_array

def create_random_initial_conditions(base_mass=1.0, base_pos_scale=1.0, base_vel_scale=0.5):
    """
    Generates randomized initial conditions for a 3-body system that are more likely to interact.
    """
    masses = [base_mass + random.uniform(-0.05, 0.05) for _ in range(3)] # Slightly varied masses
    
    positions = []
    velocities = []
    
    # Body 1 near origin with a small initial kick
    positions.append([random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)])
    velocities.append([random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)])

    # Body 2 orbiting body 1, generally in the XY plane
    r2_dist = base_pos_scale * random.uniform(0.7, 1.1) # Distance range for interaction
    theta2 = random.uniform(0, 2*np.pi)
    positions.append([r2_dist * np.cos(theta2), r2_dist * np.sin(theta2), random.uniform(-0.05, 0.05)]) # Small Z offset
    
    # Orbital velocity component for body 2 (perpendicular to position vector)
    v2_orbital_mag = base_vel_scale * np.sqrt(G_CONST * masses[0] / r2_dist) * random.uniform(0.9, 1.1) # Adjusted range
    velocities.append([-v2_orbital_mag * np.sin(theta2), v2_orbital_mag * np.cos(theta2), random.uniform(-0.01, 0.01)])

    # Body 3 interacting with the other two, potentially more elliptical
    r3_dist = base_pos_scale * random.uniform(1.2, 1.8) # Slightly further out
    theta3 = random.uniform(0, 2*np.pi)
    phi3 = random.uniform(-np.pi/6, np.pi/6) # Limit out-of-plane interaction for initial stability
    positions.append([r3_dist * np.cos(theta3) * np.cos(phi3), 
                      r3_dist * np.sin(theta3) * np.cos(phi3), 
                      r3_dist * np.sin(phi3)])

    v3_orbital_mag = base_vel_scale * np.sqrt(G_CONST * (masses[0] + masses[1]) / r3_dist) * random.uniform(0.8, 1.2) # Adjusted range
    velocities.append([-v3_orbital_mag * np.sin(theta3), v3_orbital_mag * np.cos(theta3), random.uniform(-0.01, 0.01)])


    # Center the system's center of mass velocity to zero for a stationary system CM
    total_mass = sum(masses)
    com_vel_x = sum(m * v[0] for m, v in zip(masses, velocities)) / total_mass
    com_vel_y = sum(m * v[1] for m, v in zip(masses, velocities)) / total_mass
    com_vel_z = sum(m * v[2] for m, v in zip(masses, velocities)) / total_mass

    for i in range(3):
        velocities[i][0] -= com_vel_x
        velocities[i][1] -= com_vel_y
        velocities[i][2] -= com_vel_z

    return masses, positions, velocities

# --- Main Data Generation Loop ---
if __name__ == '__main__':
    logger.info("=== DIVERSE THREE-BODY SIMULATION DATA GENERATION ===")
    
    # --- General Simulation Parameters ---
    SIM_DURATION = 80 # Significantly increased duration for more complex dynamics and interactions
    NUM_DATA_POINTS = 3000 # More snapshots per simulation
    TIME_SPAN = (0, SIM_DURATION)
    NUM_VARIANTS_PER_TYPE = 30 # Increased to 30 unique simulations for each type (more data!)
    
    # --- Generate Clean Simulations ---
    logger.info("\n--- GENERATING CLEAN SIMULATIONS ---")
    for i in range(NUM_VARIANTS_PER_TYPE):
        masses_rand, pos_rand, vel_rand = create_random_initial_conditions(
            base_pos_scale=random.uniform(0.8, 1.2), 
            base_vel_scale=random.uniform(0.4, 0.6) # Tweak these ranges to get more bound/interacting systems
        )
        
        simulate_system_robust(
            masses_rand,
            pos_rand,
            vel_rand,
            TIME_SPAN,
            NUM_DATA_POINTS,
            output_dir=os.path.join(DATA_DIR, 'clean'),
            filename_prefix=f'clean_rand_{i+1:03d}'
        )
    
    # --- Generate Perturbed Simulations ---
    logger.info("\n--- GENERATING PERTURBED SIMULATIONS ---")
    
    perturbation_types_and_ranges = {
        'hidden_mass': {
            'position_range_x': (-0.5, 0.5), 'position_range_y': (-0.5, 0.5), 'position_range_z': (-0.2, 0.2), # Smaller range for position
            'mass_range': (0.05, 0.2) # Mass of hidden body, adjusted range, smaller to avoid extreme effects
        },
        'non_inverse_square': {
            'constant_range': (1e-8, 5e-7) # Even smaller range for constant for 1/r^4 stability
        },
        'drag': {
            'coefficient_range': (0.0001, 0.0005), # Linear drag coefficient (reduced to be subtle)
            'zone_center_range_x': (-1.0, 1.0), 'zone_center_range_y': (-1.0, 1.0), 'zone_center_range_z': (-0.5, 0.5),
            'zone_radius_range': (0.5, 1.5),
            'apply_to_body': [0, 1, 2, None] # Randomly pick a body, or None for all
        },
        'impulse': {
            'time_range': (SIM_DURATION * 0.2, SIM_DURATION * 0.8), # Impulse time during simulation
            'vector_magnitude_range': (0.01, 0.05), # Magnitude of impulse vector (reduced)
            'duration_range': (0.01, 0.05), # Short duration for impulse
            'apply_to_body': [0, 1, 2] # Randomly pick a body
        },
        'time_varying_G': {
            'alpha_range': (0.0005, 0.005), # Amplitude of G variation (reduced to be subtle)
            'omega_range': (2*np.pi/50, 2*np.pi/20) # Frequency of G variation (slower cycles)
        }
    }
    
    for p_type, ranges in perturbation_types_and_ranges.items():
        logger.info(f"\n--- GENERATING {p_type.upper()} VARIANTS ---")
        for i in range(NUM_VARIANTS_PER_TYPE):
            # Use random initial conditions for each perturbed simulation as well
            masses_rand, pos_rand, vel_rand = create_random_initial_conditions(
                base_pos_scale=random.uniform(0.8, 1.2), 
                base_vel_scale=random.uniform(0.4, 0.6)
            )
            
            p_params = {}
            if p_type == 'hidden_mass':
                p_params['position'] = [random.uniform(*ranges['position_range_x']),
                                        random.uniform(*ranges['position_range_y']),
                                        random.uniform(*ranges['position_range_z'])]
                p_params['mass'] = random.uniform(*ranges['mass_range'])
            elif p_type == 'non_inverse_square':
                p_params['constant'] = random.uniform(*ranges['constant_range'])
            elif p_type == 'drag':
                p_params['coefficient'] = random.uniform(*ranges['coefficient_range'])
                p_params['zone_center'] = [random.uniform(*ranges['zone_center_range_x']),
                                           random.uniform(*ranges['zone_center_range_y']),
                                           random.uniform(*ranges['zone_center_range_z'])]
                p_params['zone_radius'] = random.uniform(*ranges['zone_radius_range'])
                p_params['body_index'] = random.choice(ranges['apply_to_body'])
            elif p_type == 'impulse':
                p_params['time'] = random.uniform(*ranges['time_range'])
                p_params['duration'] = random.uniform(*ranges['duration_range'])
                p_params['body_index'] = random.choice(ranges['apply_to_body'])
                p_params['vector'] = [random.uniform(-1, 1) * random.uniform(*ranges['vector_magnitude_range']),
                                      random.uniform(-1, 1) * random.uniform(*ranges['vector_magnitude_range']),
                                      random.uniform(-1, 1) * random.uniform(*ranges['vector_magnitude_range'])]
            elif p_type == 'time_varying_G':
                p_params['alpha'] = random.uniform(*ranges['alpha_range'])
                p_params['omega'] = random.uniform(*ranges['omega_range'])
                
            simulate_system_robust(
                masses_rand,
                pos_rand,
                vel_rand,
                TIME_SPAN,
                NUM_DATA_POINTS,
                perturbation_type=p_type,
                perturbation_params=p_params,
                output_dir=os.path.join(DATA_DIR, 'perturbed'),
                filename_prefix=f'perturbed_{p_type}_{i+1:03d}'
            )

    logger.info("\n=== ALL DATA GENERATION COMPLETE ===")
    logger.info(f"Check '{os.path.join(DATA_DIR, 'clean')}' and '{os.path.join(DATA_DIR, 'perturbed')}' for generated files.")
    logger.info("Next: Run data_analyzer.py to understand your new dataset, then proceed to model training.")