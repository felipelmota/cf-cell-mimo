import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
This function simulates the DOWNLINK performance for the three architectures.
It implements precoding schemes appropriate for downlink transmission.
"""

# --- Simulation Parameters ---

side_length = 400.0 # meters
num_realizations = 1000

# System configurations
L_distributed = 64      # APs for Cell-Free and Small Cells
M_massive_mimo = 64     # Antennas for Cellular Massive MIMO
K = 8                   # Number of UEs in all setups

# Bandwidth for rate calculation
B = 20e6  # 20 MHz

# Downlink power per AP and Noise power at UE
p_ap_dbm = 23.0  # 200 mW
p_ap_mW = 10**(p_ap_dbm / 10)
noise_power_dbm = -96.0
noise_power_mW = 10**(noise_power_dbm / 10)

# Power for channel estimation to design the precoders
p_ue_mW_pilot = 100.0

ap_height = 10.0

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
    # Use pseudo-inverse for better stability if matrix is singular
    uplink_covariance_inv = np.linalg.pinv(uplink_covariance)
    V_mmse_unscaled = uplink_covariance_inv @ G_ul.conj().T
    
    normalization_factor = np.linalg.norm(V_mmse_unscaled, axis=0, keepdims=True)
    V_normalized = V_mmse_unscaled / normalization_factor
    return V_normalized

# --- AP/BS Positions ---
ap_pos_massive_mimo = np.array([[side_length / 2, side_length / 2]])
grid_points = np.linspace(side_length / 16, side_length * 15 / 16, 8)
x_coords, y_coords = np.meshgrid(grid_points, grid_points)
ap_pos_distributed = np.vstack([x_coords.ravel(), y_coords.ravel()]).T

# --- Store simulation results ---
all_rates_cf = []
all_rates_sc = []
all_rates_mmimo = []

# Data for scatter plots
scatter_data = {}

print("Running Extended DOWNLINK simulation...")
for i in range(num_realizations):
    ue_pos = np.random.uniform(0, side_length, size=(K, 2))

    gains_db_mmimo = get_channel_gains_db(ue_pos, ap_pos_massive_mimo)
    gains_linear_mmimo = 10**(gains_db_mmimo / 10)
    random_phases_mmimo = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=(K, M_massive_mimo)))
    G_ul = np.sqrt(gains_linear_mmimo) * random_phases_mmimo
    
    gains_db_dist = get_channel_gains_db(ue_pos, ap_pos_distributed)
    gains_linear_dist = 10**(gains_db_dist / 10)
    random_phases_dist = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=(K, L_distributed)))
    H_ul = np.sqrt(gains_linear_dist) * random_phases_dist
    
    rates_this_run_mmimo = []
    rates_this_run_cf = []
    rates_this_run_sc = []

    # --- SINR and Rate Calculation ---
    
    # 1. Cellular: Massive MIMO
    V_normalized = calculate_mmse_precoding_vectors(p_ue_mW_pilot, noise_power_mW, G_ul)
    V_precoding = np.sqrt(p_ap_mW / K) * V_normalized
    
    for k in range(K):
        g_k = G_ul[k, :]
        v_k = V_precoding[:, k]
        signal_power = np.abs(g_k @ v_k)**2
        interference_indices = np.delete(np.arange(K), k)
        interference_power = np.sum(np.abs(g_k @ V_precoding[:, interference_indices])**2)
        sinr = signal_power / (interference_power + noise_power_mW)
        rate = B * np.log2(1 + sinr) / 1e6 # in Mbps
        all_rates_mmimo.append(rate)
        rates_this_run_mmimo.append(rate)

    # 2. Cell-Free
    W_normalized = calculate_mmse_precoding_vectors(p_ue_mW_pilot, noise_power_mW, H_ul)
    # Per-AP power normalization
    power_per_user_per_ap = p_ap_mW / K
    normalization_per_ap = np.sqrt(power_per_user_per_ap / np.mean(np.abs(W_normalized)**2, axis=0, keepdims=True))
    W_precoding = W_normalized * normalization_per_ap
    
    for k in range(K):
        h_k = H_ul[k, :]
        w_k = W_precoding[:, k]
        signal_power = np.abs(h_k @ w_k)**2
        interference_indices = np.delete(np.arange(K), k)
        interference_power = np.sum(np.abs(h_k @ W_precoding[:, interference_indices])**2)
        sinr = signal_power / (interference_power + noise_power_mW)
        rate = B * np.log2(1 + sinr) / 1e6 # in Mbps
        all_rates_cf.append(rate)
        rates_this_run_cf.append(rate)

    # 3. Cellular: Small Cells
    channel_gains_sq = np.abs(H_ul)**2
    serving_ap_indices = np.argmax(channel_gains_sq, axis=1)
    
    for k in range(K):
        serving_ap = serving_ap_indices[k]
        signal_power = p_ap_mW * channel_gains_sq[k, serving_ap]
        interference_power = 0
        for l in range(L_distributed):
            if l != serving_ap:
                interference_power += p_ap_mW * channel_gains_sq[k, l]
        sinr = signal_power / (interference_power + noise_power_mW)
        rate = B * np.log2(1 + sinr) / 1e6 # in Mbps
        all_rates_sc.append(rate)
        rates_this_run_sc.append(rate)

        scatter_data['ue_pos'] = ue_pos
        scatter_data['rates_mmimo'] = np.array(rates_this_run_mmimo)
        scatter_data['rates_cf'] = np.array(rates_this_run_cf)


print("Plotting results...")

# --- Plotting ---
labels = ['Cell-Free', 'Cellular: Small cells', 'Cellular: Massive MIMO']
colors = ['blue', 'red', 'black']

# Average Rate Bar Chart
avg_rates = [np.mean(all_rates_cf), np.mean(all_rates_sc), np.mean(all_rates_mmimo)]
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, avg_rates, color=colors)
plt.ylabel('Average Rate [Mbps]', fontsize=12)
plt.title('Average Per-User Downlink Rate Comparison', fontsize=14)
plt.bar_label(bars, fmt='%.1f')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Median Rate Bar Chart
median_rates = [np.median(all_rates_cf), np.median(all_rates_sc), np.median(all_rates_mmimo)]
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, median_rates, color=colors)
plt.ylabel('Median Rate [Mbps]', fontsize=12)
plt.title('Median Per-User Downlink Rate Comparison (Fairness Metric)', fontsize=14)
plt.bar_label(bars, fmt='%.1f')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
