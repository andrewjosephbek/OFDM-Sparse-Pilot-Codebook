import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from remoteRF.drivers.adalm_pluto import *


def init_sdr(fs, fc, tx_gain_dB, rx_gain_dB, token):
    sdr = adi.Pluto(token=token)
    sdr.sample_rate = int(fs)

    # TX config
    sdr.tx_destroy_buffer()
    sdr.tx_lo = int(fc)
    sdr.tx_rf_bandwidth = int(fs)
    sdr.tx_hardwaregain_chan0 = tx_gain_dB
    sdr.tx_cyclic_buffer = True    # repeat entire frame continuously

    # RX config (except buffer size)
    sdr.rx_destroy_buffer()
    sdr.rx_lo = int(fc)
    sdr.rx_rf_bandwidth = int(fs)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = rx_gain_dB

    return sdr

def generate_qam_constellation(M):
    dim = int(np.sqrt(M))
    if dim * dim != M:
        raise ValueError("M must be a perfect square (e.g., 16, 64, 256, 1024)")
    
    levels = np.arange(-(dim-1), dim, 2)
    I, Q = np.meshgrid(levels, levels)
    QAMM = (I + 1j * Q).flatten()

    # Normalize to unit average power
    QAMM = QAMM / np.sqrt(np.mean(np.abs(QAMM)**2))
    
    return QAMM

def build_preamble(N_fft, N_cp, active_subcar, QPSK):
    preamble_fd = np.zeros(N_fft, dtype=complex)
    preamble_fd[active_subcar] = np.random.choice(QPSK, size=len(active_subcar))

    preamble_td = np.fft.ifft(np.fft.ifftshift(preamble_fd))
    preamble_td = np.concatenate([preamble_td[-N_cp:], preamble_td])

    preamble = np.concatenate([preamble_td, preamble_td])  # repeat twice
    return preamble


def build_payload(N_sym, N_fft, N_cp, OFDM_CSTL, dead_subcar, pilot_subcar, pilot_symbol):

    # (1) Random OFDM symbols for all OFDM symbols at once
    M = np.random.choice(OFDM_CSTL, size=(N_sym, N_fft))

    # (2) Apply DC null + pilots (fully vectorized)
    M[:, dead_subcar] = 0
    M[:, pilot_subcar] = pilot_symbol

    # (3) IFFT shift all rows at once
    M_unshift = np.fft.ifftshift(M, axes=1)

    # (4) Compute IFFT for all OFDM symbols simultaneously
    X_no_cp = np.fft.ifft(M_unshift, n=N_fft, axis=1)

    # (5) Add CP vectorized
    cp = X_no_cp[:, -N_cp:]                    # shape (N_sym, N_cp)
    X_with_cp = np.concatenate([cp, X_no_cp], axis=1)   # shape (N_sym, N_fft+N_cp)

    # (6) Flatten into single time-domain sequence
    X = X_with_cp.reshape(-1)

    return M, X



def detect_preamble(rx, preamble, full_frame_len):
    search_len = min(2 * full_frame_len, len(rx))
    corr = correlate(rx[:search_len], preamble, mode="full")
    corr_abs = np.abs(corr)

    thresh = 0.80 * np.max(corr_abs)
    peak_candidates = np.where(corr_abs > thresh)[0]
    if len(peak_candidates) == 0:
        raise RuntimeError("No preamble detected!")

    peak_index = peak_candidates[0]
    start_index = peak_index - (len(preamble) - 1)
    return max(start_index, 0)


def extract_payload(rx, start_index, preamble_len, payload_len):
    L = len(rx)
    payload_start = start_index + preamble_len
    payload_end = payload_start + payload_len

    if payload_end > L:
        print("WARNING: RX capture too short, padding zeros")
        shortage = payload_end - L
        rx_payload = np.concatenate([rx[payload_start:L],
                                     np.zeros(shortage, dtype=complex)])
    else:
        rx_payload = rx[payload_start:payload_end]

    return rx_payload


def estimate_channel(Y, pilot_subcar, pilot_symbol, N_fft):
    Y_pilots = Y[:, pilot_subcar]
    H_pilots_all = Y_pilots / pilot_symbol
    H_pilots = H_pilots_all.mean(axis=0)

    sub_idx = np.arange(N_fft)
    pilot_idx = pilot_subcar
    # H_real = np.interp(sub_idx, pilot_idx, H_pilots.real)
    # H_imag = np.interp(sub_idx, pilot_idx, H_pilots.imag)
    # H_est_full = H_real + 1j * H_imag
    H_mag = np.interp(sub_idx, pilot_idx, np.abs(H_pilots))
    H_phase = np.interp(sub_idx, pilot_idx, np.angle(H_pilots))
    H_est_full = H_mag * np.exp(1j * H_phase)

    # Noise power = channel power variance 
    noise = H_pilots_all - H_pilots[None, :]
    noise_power = np.mean(np.abs(noise)**2)

    signal_power = np.mean(np.abs(H_pilots)**2)

    SNR_linear = signal_power / noise_power
    SNR_dB = 10*np.log10(SNR_linear)

    # print(f"Estimated SNR: {SNR_dB:.2f} dB")

    return SNR_dB, H_est_full


def equalize_and_ser(M, data_subcar, Y, H_est_full, OFDM_CSTL):
    eps = 1e-8
    H_safe = np.copy(H_est_full)
    H_safe[np.abs(H_safe) < eps] = eps

    Y_eq = Y / H_safe[None, :]

    M_data = M[:, data_subcar]
    Y_data = Y_eq[:, data_subcar]

    tx_flat = M_data.flatten()
    rx_flat = Y_data.flatten()

    # Normalize average energy of received constellation
    r = rx_flat / np.sqrt(np.mean(np.abs(rx_flat)**2))

    # Nearest-neighbor detection
    dists = np.abs(r[:, None] - OFDM_CSTL[None, :])
    rx_dec = OFDM_CSTL[np.argmin(dists, axis=1)]

    symbol_errors = np.sum(rx_dec != tx_flat)
    SER = symbol_errors / len(tx_flat)

    return SER, r


#### Main

### SDR Params 
fs = 1e6           # baseband sample rate
fc = 915e6         # RF center frequency
tx_gain_dB = -20
rx_gain_dB = 40
token = "EnNuEIVGpfc" 

### OFDM Params

# OFDM payload parameters
N_sym = 200
N_fft = 256 + 1 # must be odd
N_cp = 6
pilot_density = 16
N_guard = 16

# total transmission payload length in samples
payload_len = (N_fft + N_cp) * N_sym

# Constellations
QPSK = np.array([
    (1+1j)/np.sqrt(2),
    (-1+1j)/np.sqrt(2),
    (-1-1j)/np.sqrt(2),
    (1-1j)/np.sqrt(2)
], dtype=complex)

QAM16 = generate_qam_constellation(16)

OFDM_CSTL = QAM16.copy()
pilot_symbol = (1+1j)/np.sqrt(2)

# DC subcarrier index
dc_idx = N_fft // 2

# Subcarrier assignments 
subcars = np.arange(N_fft)
dead_subcar = np.array([dc_idx])
dead_subcar = np.append(dead_subcar, [subcars[:N_guard], subcars[-N_guard:]])

pilot_subcar = np.arange(0, N_fft, pilot_density)
pilot_subcar = pilot_subcar[pilot_subcar != dc_idx]

# Estimate the whole channel
# pilot_subcar = subcars
# pilot_subcar = pilot_subcar[pilot_subcar != dc_idx]
# pilot_subcar = pilot_subcar[pilot_subcar != dc_idx+5]
# pilot_subcar = pilot_subcar[pilot_subcar != dc_idx-5]


active_subcar = subcars[~np.isin(subcars, dead_subcar)]
data_subcar = subcars[~np.isin(subcars, np.concatenate((pilot_subcar, dead_subcar)))]

print(f"BW:             {fs/1e6} MHz")
print(f"Carrier Freq:   {fc/1e6} MHz")
print(f"Subcarriers:    {N_fft}")
print(f"Pilot Spacing:  {int(np.ceil(N_fft/len(pilot_subcar)))} subcarriers")
print(f"Cyclic Prefix:  {N_cp} samples")
print(f"QAM Order:      {len(OFDM_CSTL)}")

### Build OFDM-style preamble
preamble = build_preamble(N_fft, N_cp, active_subcar, QPSK)
preamble_len = len(preamble)

### Build OFDM payload
M, payload = build_payload(N_sym, N_fft, N_cp,
                           OFDM_CSTL, dead_subcar,
                           pilot_subcar, pilot_symbol)

full_frame_len = payload_len + preamble_len

### Initialize SDR (buffer size set AFTER we know full_frame_len)
sdr = init_sdr(fs, fc, tx_gain_dB, rx_gain_dB, token)
sdr.rx_buffer_size = int(5 * full_frame_len)

### Build full TX frame
tx_frame = np.concatenate([preamble, payload])

# Scale for Pluto DAC
tx_scaled = tx_frame / np.max(np.abs(tx_frame)) * (2**14)

### Transmit
sdr.tx(tx_scaled)
print("Transmitting ...")

### Receive and flush old samples
for _ in range(3):
    _ = sdr.rx()

rx = sdr.rx()
L = len(rx)

### Correlation with OFDM preamble
start_index = detect_preamble(rx, preamble, full_frame_len)
print("Detected start_index:", start_index)

### Extract payload
rx_payload = extract_payload(rx, start_index, preamble_len, payload_len)

### Reshape and remove CP
rx_mat = rx_payload.reshape(N_sym, N_fft + N_cp)
rx_no_cp = rx_mat[:, N_cp:]

### FFT to construct freq domain signal
Y_unshifted = np.fft.fft(rx_no_cp, axis=1)
Y = np.fft.fftshift(Y_unshifted, axes=1)

### Channel estimation from pilots
SNR_dB, H_est_full = estimate_channel(Y, pilot_subcar, pilot_symbol, N_fft)

### 1-tap equalization and SER
SER, rx_flat = equalize_and_ser(M, data_subcar, Y, H_est_full, OFDM_CSTL)

print(f"Total symbols:      {len(M[:, data_subcar].flatten())}")
print(f"SER:                {SER*100:.4f} %")

### Plots
fig, axs = plt.subplots(1, 2, figsize=(10, 6))

x = np.linspace(-500, 500, N_fft)

# |H| over subcarriers
axs[0].plot(x, np.abs(H_est_full))

# highlight pilot subcarriers
pilot_freqs = x[pilot_subcar]
axs[0].scatter(
    pilot_freqs,
    np.abs(H_est_full[pilot_subcar]),
    color="black",
    s=20,
    label="pilots"
)

axs[0].set_title("|H| over Subcarriers")
axs[0].set_xlabel("f (kHz)")
axs[0].set_ylabel("|H|")
axs[0].grid(True)

# Constellation Plot
axs[1].scatter(rx_flat.real, rx_flat.imag, s=4, color="black", label="RX")
axs[1].scatter(OFDM_CSTL.real, OFDM_CSTL.imag, s=40, color="red", marker="x", label="Ideal")
axs[1].set_title(f"Equalized Constellation (M = {len(OFDM_CSTL)}, Nc = {len(active_subcar)})")
axs[1].set_xlabel("I")
axs[1].set_ylabel("Q")
axs[1].grid(True)
axs[1].set_aspect('equal', 'box')
axs[1].legend(loc="upper right", frameon=True)


plt.tight_layout()
plt.show()
