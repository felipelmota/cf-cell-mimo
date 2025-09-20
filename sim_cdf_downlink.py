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

# Downlink power per AP and Noise power at UE
p_ap_dbm = 23.0  # 200 mW
p_ap_mW = 10**(p_ap_dbm / 10)
noise_power_dbm = -96.0
noise_power_mW = 10**(noise_power_dbm / 10)

# Power for channel estimation to design the precoders
p_ue_mW = 100.0

ap_height = 10.0 # meters

# --- Helper Functions ---
def pathloss_db(d):
    return -30.5 - 36.7 * np.log10(d + 1e-9)

def get_channel_gains_db(ue_pos, ap_pos):
    """ Calculates large-scale fading based on distance """
    # Calculate distance between each UE and each AP
    distances = np.sqrt(
        np.sum((ue_pos[:, np.newaxis, :] - ap_pos[np.newaxis, :, :])**2, axis=2) 
        + ap_height**2
    )
    return pathloss_db(distances)

# --- AP/BS Positions ---
ap_pos_massive_mimo = np.array([[side_length / 2, side_length / 2]])
grid_points = np.linspace(side_length / 16, side_length * 15 / 16, 8)
x_coords, y_coords = np.meshgrid(grid_points, grid_points)
ap_pos_distributed = np.vstack([x_coords.ravel(), y_coords.ravel()]).T

# --- Store SINR results ---
all_sinr_cf = []
all_sinr_sc = []
all_sinr_mmimo = []

print("Running DOWNLINK simulation...")
for i in range(num_realizations):
    ue_pos = np.random.uniform(0, side_length, size=(K, 2))

    # --- Generate Channels for each system ---

    # 1. Channels for Massive MIMO
    gains_db_mmimo = get_channel_gains_db(ue_pos, ap_pos_massive_mimo)
    gains_linear_mmimo = 10**(gains_db_mmimo / 10)
    random_phases_mmimo = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=(K, M_massive_mimo)))
    G_ul = np.sqrt(gains_linear_mmimo) * random_phases_mmimo

    # 2. Channels for Distributed Systems (Cell-Free & Small Cells)
    gains_db_dist = get_channel_gains_db(ue_pos, ap_pos_distributed)
    gains_linear_dist = 10**(gains_db_dist / 10)
    random_phases_dist = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=(K, L_distributed)))
    H_ul = np.sqrt(gains_linear_dist) * random_phases_dist
    
    # --- Precoding and SINR Calculation ---
    
    # 1. Cellular: Massive MIMO
    # Calculate MMSE precoding vectors based on uplink channel estimates
    uplink_covariance = p_ue_mW * (G_ul.conj().T @ G_ul) + noise_power_mW * np.eye(M_massive_mimo)
    V_mmse_unscaled = np.linalg.solve(uplink_covariance, G_ul.conj().T)
    
    # Normalize precoding vectors and apply equal power allocation
    normalization_factor = np.linalg.norm(V_mmse_unscaled, axis=0)
    V_precoding = np.sqrt(p_ap_mW / K) * V_mmse_unscaled / normalization_factor
    
    for k in range(K):
        g_k = G_ul[k, :]
        v_k = V_precoding[:, k]

        signal_power = np.abs(g_k @ v_k)**2

        interference_indices = np.delete(np.arange(K), k)
        interference_power = np.sum(np.abs(g_k @ V_precoding[:, interference_indices])**2)

        sinr = signal_power / (interference_power + noise_power_mW)
        all_sinr_mmimo.append(sinr)

    # 2. Cell-Free
    # Calculate MMSE precoding vectors for the distributed system
    uplink_covariance_cf = p_ue_mW * (H_ul.conj().T @ H_ul) + noise_power_mW * np.eye(L_distributed)
    W_mmse_unscaled = np.linalg.solve(uplink_covariance_cf, H_ul.conj().T)

    # Normalize power to satisfy per-AP power constraints
    power_per_user_per_ap = p_ap_mW / K
    normalization_per_ap = np.sqrt(power_per_user_per_ap / np.mean(np.abs(W_mmse_unscaled)**2, axis=0))
    W_precoding = W_mmse_unscaled * normalization_per_ap
    
    for k in range(K):
        h_k = H_ul[k, :]
        w_k = W_precoding[:, k]

        signal_power = np.abs(h_k @ w_k)**2

        interference_indices = np.delete(np.arange(K), k)
        interference_power = np.sum(np.abs(h_k @ W_precoding[:, interference_indices])**2)

        sinr = signal_power / (interference_power + noise_power_mW)
        all_sinr_cf.append(sinr)

    # 3. Cellular: Small Cells
    # Find AP for each user
    channel_gains_sq = np.abs(H_ul)**2
    available_ap_indices = np.argmax(channel_gains_sq, axis=1)

    for k in range(K):
        available_ap = available_ap_indices[k]
        h_kl_2 = channel_gains_sq[k, available_ap]

        signal_power = p_ap_mW * h_kl_2

        # Interference from all other APs serving their respective users
        interference_power = 0
        for l in range(L_distributed):
            if l != available_ap:
                # Assume other APs are serving their best user, which is a source of interference
                interference_power += p_ap_mW * channel_gains_sq[k, l]

        sinr = signal_power / (interference_power + noise_power_mW)
        all_sinr_sc.append(sinr)

print("Simulation finished. Plotting results...")

# --- Plotting ---
plt.style.use('default')
plt.figure(figsize=(10, 7))

sns.ecdfplot(10 * np.log10(all_sinr_cf), 
                label='Cell-free', 
                color='blue', 
                linestyle='--',
                linewidth=2.5)
sns.ecdfplot(10 * np.log10(all_sinr_sc), 
                label='Cellular: Small cells', 
                color='red', 
                linestyle='-.',
                linewidth=2.5)
sns.ecdfplot(10 * np.log10(all_sinr_mmimo), 
                label='Cellular: Massive MIMO', 
                color='black', 
                linestyle='-',
                linewidth=2.5)

plt.axhline(0.05, color='gray', linestyle='--', linewidth=1)
plt.text(2, 0.07, '95% likely', fontsize=14, color='black', ha='left')

plt.xlabel('SINR [dB]', fontsize=16)
plt.ylabel('CDF', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(-15, 60)
plt.ylim(0, 1)
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.legend(fontsize=14)
plt.title('SINR Over Cellular Architectures - Downlink', fontsize=18)
plt.show()
