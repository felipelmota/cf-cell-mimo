import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --- Simulation Parameters ---
side_length = 400.0  # meters
num_realizations = 1000

# System configurations
L_distributed = 64      # APs for Cell-Free and Small Cells
M_massive_mimo = 64     # Antennas for Cellular Massive MIMO
K = 8                   # Number of UEs in all setups

# Power and Noise
p_dl_dbm = 10.0
p_dl_mW = 10**(p_dl_dbm / 10)
noise_power_dbm = -96.0
noise_power_mW = 10**(noise_power_dbm / 10)

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

def calculate_mmse_sinr(p, noise, h_k, H_interf):
    """
    Calculates SINR using the matrix inversion lemma for stability.
    SINR = p * h_k^H * (p * H_interf * H_interf^H + noise * I)^-1 * h_k

    Args:
        p (float): Transmit power in mW.
        noise (float): Noise power in mW.
        h_k (np.array): Channel vector of the desired user (N x 1).
        H_interf (np.array): Channel matrix of interfering users (N x K-1).
    """
    N, K_interf = H_interf.shape

    # Small (K-1)x(K-1) matrix to invert
    small_matrix_inv = np.linalg.inv(
        np.eye(K_interf) + (p / noise) * H_interf.conj().T @ H_interf
    )

    # Apply the matrix inversion lemma
    # (A+UCV)^-1 = A^-1 - A^-1 U (C^-1 + V A^-1 U)^-1 V A^-1
    term1 = (1 / noise) * h_k
    term2 = (p / noise**2) * (H_interf @ (small_matrix_inv @ (H_interf.conj().T @ h_k)))

    sinr = p * np.abs(h_k.conj().T @ (term1 - term2))
    return sinr.item()


# --- AP/BS Positions ---
ap_pos_massive_mimo = np.array([[side_length / 2, side_length / 2]])
grid_points = np.linspace(side_length / 16, side_length * 15 / 16, 8)
x_coords, y_coords = np.meshgrid(grid_points, grid_points)
ap_pos_distributed = np.vstack([x_coords.ravel(), y_coords.ravel()]).T

# --- Store SINR results ---
all_sinr_cf = []
all_sinr_sc = []
all_sinr_mmimo = []

print("Running UPLINK simulation...")
for i in range(num_realizations):
    ue_pos = np.random.uniform(0, side_length, size=(K, 2))

    # --- Generate Channels for each system ---

    # 1. Channels for Massive MIMO
    gains_db_mmimo = get_channel_gains_db(ue_pos, ap_pos_massive_mimo)
    gains_linear_mmimo = 10**(gains_db_mmimo / 10)
    random_phases_mmimo = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=(K, M_massive_mimo)))
    G = np.sqrt(gains_linear_mmimo) * random_phases_mmimo

    # 2. Channels for Distributed Systems (Cell-Free & Small Cells)
    gains_db_dist = get_channel_gains_db(ue_pos, ap_pos_distributed)
    gains_linear_dist = 10**(gains_db_dist / 10)
    random_phases_dist = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=(K, L_distributed)))
    H = np.sqrt(gains_linear_dist) * random_phases_dist

    # --- SINR Calculation ---

    # Cell-Free
    for k in range(K):
        h_k = H[k, :].reshape(L_distributed, 1)
        H_interf = np.delete(H, k, axis=0).T
        sinr = calculate_mmse_sinr(p_dl_mW, noise_power_mW, h_k, H_interf)
        all_sinr_cf.append(sinr)

    # Cellular: Small Cells
    for k in range(K):
        h_kl_2 = np.abs(H[k, :])**2
        signal_power = p_dl_mW * h_kl_2
        interference_power = p_dl_mW * (np.sum(np.abs(H)**2, axis=0) - h_kl_2)
        sinr_per_ap = signal_power / (interference_power + noise_power_mW)
        all_sinr_sc.append(np.max(sinr_per_ap))

    # Cellular: Massive MIMO
    for k in range(K):
        g_k = G[k, :].reshape(M_massive_mimo, 1)
        G_interf = np.delete(G, k, axis=0).T
        sinr = calculate_mmse_sinr(p_dl_mW, noise_power_mW, g_k, G_interf)
        all_sinr_mmimo.append(sinr)

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
plt.xlim(0, 60)
plt.ylim(0, 1)
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.legend(fontsize=14)
plt.title('SINR Over Cellular Architectures - Uplink', fontsize=18)
plt.show()
