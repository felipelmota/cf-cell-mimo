import numpy as np
import matplotlib.pyplot as plt

"""
This function generates two high-density scatter plots for downlink user rates,
one for Cellular Massive MIMO and one for Cell-Free Massive MIMO,
replicating the visual style of the user-provided examples.
"""

# --- Simulation Parameters ---
side_length = 400.0
num_users_to_plot = 1000 # Number of points in the scatter plot

# System infrastructure
L_distributed = 64      # APs for Cell-Free
M_massive_mimo = 64     # Antennas for Cellular Massive MIMO
K = 8

# Bandwidth for rate calculation
B = 20e6  # 20 MHz

# Power and Noise
p_ap_dbm = 23.0  # 200 mW
p_ap_mW = 10**(p_ap_dbm / 10)
noise_power_dbm = -96.0
noise_power_mW = 10**(noise_power_dbm / 10)
ap_height = 10.0
p_ue_mW_pilot = 100.0

# --- Helper Functions ---
def pathloss_db(d):
    return -30.5 - 36.7 * np.log10(d + 1e-9)

def get_channel_gains_db(ue_pos, ap_pos):
    distances = np.sqrt(
        np.sum((ue_pos[:, np.newaxis, :] - ap_pos[np.newaxis, :, :])**2, axis=2) 
        + ap_height**2
    )
    return pathloss_db(distances)

def calculate_mmse_precoding_vectors(p_pilot, noise, G_ul):
    N_antennas = G_ul.shape[1]
    uplink_covariance = p_pilot * (G_ul.conj().T @ G_ul) + noise * np.eye(N_antennas)
    uplink_covariance_inv = np.linalg.pinv(uplink_covariance) # Use pseudo-inverse for stability
    V_mmse_unscaled = uplink_covariance_inv @ G_ul.conj().T
    normalization_factor = np.linalg.norm(V_mmse_unscaled, axis=0, keepdims=True)
    V_normalized = V_mmse_unscaled / (normalization_factor + 1e-9) # adding to avoid div by zero
    return V_normalized

# --- AP/BS Positions ---
ap_pos_massive_mimo = np.array([[side_length / 2, side_length / 2]])
grid_points = np.linspace(side_length / 16, side_length * 15 / 16, 8)
x_coords, y_coords = np.meshgrid(grid_points, grid_points)
ap_pos_distributed = np.vstack([x_coords.ravel(), y_coords.ravel()]).T

# --- Generate all user positions beforehand ---
all_user_positions = np.random.uniform(0, side_length, size=(num_users_to_plot, 2))

# --- Store final rate results for plotting ---
rates_cf = []
rates_mmimo = []

print(f"Running massive scatter plot simulation for {num_users_to_plot} users...")
for i in range(num_users_to_plot):
    # The user we are currently evaluating
    test_user_pos = all_user_positions[i].reshape(1, 2)
    # Generate K-1 random interferers for this user's simulation
    interferer_pos = np.random.uniform(0, side_length, size=(K - 1, 2))

    current_ue_pos = np.vstack([test_user_pos, interferer_pos])

    # --- Generate Channels for this specific scenario ---
    gains_db_mmimo = get_channel_gains_db(current_ue_pos, ap_pos_massive_mimo)
    gains_linear_mmimo = 10**(gains_db_mmimo / 10)
    random_phases_mmimo = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=(K, M_massive_mimo)))
    G_ul = np.sqrt(gains_linear_mmimo) * random_phases_mmimo
    
    gains_db_dist = get_channel_gains_db(current_ue_pos, ap_pos_distributed)
    gains_linear_dist = 10**(gains_db_dist / 10)
    random_phases_dist = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=(K, L_distributed)))
    H_ul = np.sqrt(gains_linear_dist) * random_phases_dist
    
    # --- Cellular: Massive MIMO Calculation ---
    V_normalized = calculate_mmse_precoding_vectors(p_ue_mW_pilot, noise_power_mW, G_ul)
    V_precoding = np.sqrt(p_ap_mW / K) * V_normalized
    g_k = G_ul[0, :] # Channel for our test user (index 0)
    v_k = V_precoding[:, 0]
    signal_power_mmimo = np.abs(g_k @ v_k)**2
    interference_power_mmimo = np.sum(np.abs(g_k @ V_precoding[:, 1:])**2)
    sinr_mmimo = signal_power_mmimo / (interference_power_mmimo + noise_power_mW)
    rate_mmimo = B * np.log2(1 + sinr_mmimo) / 1e6
    rates_mmimo.append(rate_mmimo)

    # --- Cell-Free Calculation ---
    W_normalized = calculate_mmse_precoding_vectors(p_ue_mW_pilot, noise_power_mW, H_ul)
    power_per_user_per_ap = p_ap_mW / K
    normalization_per_ap = np.sqrt(power_per_user_per_ap / np.mean(np.abs(W_normalized)**2, axis=0, keepdims=True))
    W_precoding = W_normalized * (normalization_per_ap + 1e-9)
    h_k = H_ul[0, :] # Channel for our test user (index 0)
    w_k = W_precoding[:, 0]
    signal_power_cf = np.abs(h_k @ w_k)**2
    interference_power_cf = np.sum(np.abs(h_k @ W_precoding[:, 1:])**2)
    sinr_cf = signal_power_cf / (interference_power_cf + noise_power_mW)
    rate_cf = B * np.log2(1 + sinr_cf) / 1e6
    rates_cf.append(rate_cf)

print("Plotting results...")

# --- Plotting ---

# 1. Scatter Plot for Cellular Massive MIMO
plt.figure(figsize=(10, 8))
scatter = plt.scatter(all_user_positions[:, 0], all_user_positions[:, 1], 
                        c=rates_mmimo, cmap='viridis_r', s=50, alpha=0.9,
                        edgecolors='black', linewidth=0.5)
plt.title(f'User Rate Distribution in Cellular Massive MIMO', fontsize=16)
plt.xlabel('X (m)', fontsize=12)
plt.ylabel('Y (m)', fontsize=12)
plt.legend()
cbar = plt.colorbar(scatter, fraction=0.046, pad=0.04)
cbar.set_label('User Rate (Mbps)', fontsize=12)
plt.xlim(0, side_length)
plt.ylim(0, side_length)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# 2. Scatter Plot for Cell-Free Massive MIMO
plt.figure(figsize=(10, 8))
scatter = plt.scatter(all_user_positions[:, 0], all_user_positions[:, 1], 
                        c=rates_cf, cmap='viridis_r', s=50, alpha=0.9,
                        edgecolors='black', linewidth=0.5)
plt.title(f'User Rate Distribution in Cell-Free Massive MIMO', fontsize=16)
plt.xlabel('X (m)', fontsize=12)
plt.ylabel('Y (m)', fontsize=12)
plt.legend()
cbar = plt.colorbar(scatter, fraction=0.046, pad=0.04)
cbar.set_label('User Rate (Mbps)', fontsize=12)
plt.xlim(0, side_length)
plt.ylim(0, side_length)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

