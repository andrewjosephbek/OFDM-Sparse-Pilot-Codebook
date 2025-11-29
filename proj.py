import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from remoteRF.drivers.adalm_pluto import *


def init_sdr(fs, fc, tx_gain_dB, rx_gain_dB, token):
    """Initialize Pluto SDR with basic TX/RX settings (no buffer size here)."""
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


def build_preamble(N_fft, N_cp, active_subcar, QPSK):
    """Build complex OFDM-style preamble with CP, repeated twice."""
    preamble_fd = np.zeros(N_fft, dtype=complex)
    preamble_fd[active_subcar] = np.random.choice(QPSK, size=len(active_subcar))

    preamble_td = np.fft.ifft(np.fft.ifftshift(preamble_fd))
    preamble_td = np.concatenate([preamble_td[-N_cp:], preamble_td])

    preamble = np.concatenate([preamble_td, preamble_td])  # repeat twice
    return preamble


def build_payload(N_sym, N_fft, N_cp, OFDM_CSTL, dead_subcar, pilot_subcar, pilot_symbol):
    """
    Build frequency-domain symbol matrix M and time-domain payload X
    including CP, one OFDM symbol per row of M.
    """
    # Random data symbols
    M = np.random.choice(OFDM_CSTL, size=(N_sym, N_fft))

    # Force DC off and set pilots
    M[:, dead_subcar] = 0
    M[:, pilot_subcar] = pilot_symbol

    # Time-domain OFDM with CP
    X = np.zeros((N_fft + N_cp) * N_sym, dtype=complex)
    idx = 0

    for s in range(N_sym):
        xf = M[s, :]
        xf_unshift = np.fft.ifftshift(xf)
        xt = np.fft.ifft(xf_unshift, n=N_fft)
        xt_cp = np.concatenate([xt[-N_cp:], xt])
        X[idx: idx + N_fft + N_cp] = xt_cp
        idx += N_fft + N_cp

    return M, X


def detect_preamble(rx, preamble, full_frame_len):
    """Find preamble via correlation and return estimated start index."""
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
    """Slice the RX buffer to get the payload, padding with zeros if needed."""
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
    """1-tap channel estimation on pilots, then linear interpolation over all subcarriers."""
    Y_pilots = Y[:, pilot_subcar]
    H_pilots_all = Y_pilots / pilot_symbol
    H_pilots = H_pilots_all.mean(axis=0)

    sub_idx = np.arange(N_fft)
    pilot_idx = pilot_subcar
    H_real = np.interp(sub_idx, pilot_idx, H_pilots.real)
    H_imag = np.interp(sub_idx, pilot_idx, H_pilots.imag)
    H_est_full = H_real + 1j * H_imag
    return H_est_full


def equalize_and_ser(M, data_subcar, Y, H_est_full, OFDM_CSTL):
    """Apply 1-tap equalization, detect QAM symbols, and compute SER."""
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

    return SER, rx_flat


# =========================================================
#                Main Script
# =========================================================

### SDR Params 
fs = 1e6           # baseband sample rate
fc = 915e6         # RF center frequency
tx_gain_dB = -20
rx_gain_dB = 40
token = "sCBx5l91z7I" 

### OFDM Params

# OFDM payload parameters
N_sym = 200
N_fft = 256 + 1   # 257 subcarriers
N_cp = 16

# total transmission payload length in samples
payload_len = (N_fft + N_cp) * N_sym

# Constellations
QPSK = np.array([
    (1+1j)/np.sqrt(2),
    (-1+1j)/np.sqrt(2),
    (-1-1j)/np.sqrt(2),
    (1-1j)/np.sqrt(2)
], dtype=complex)

QAM16 = np.array([
    -3-3j, -3-1j, -3+1j, -3+3j,
    -1-3j, -1-1j, -1+1j, -1+3j,
     1-3j,  1-1j,  1+1j,  1+3j,
     3-3j,  3-1j,  3+1j,  3+3j
], dtype=complex)

# Normalize to unit average power
QAM16 /= np.sqrt((np.abs(QAM16)**2).mean())

OFDM_CSTL = QAM16.copy()
pilot_symbol = (1+1j)/np.sqrt(2)

# DC subcarrier index
dc_idx = N_fft // 2

# Subcarrier assignments 
subcars = np.arange(N_fft)
dead_subcar = np.array([dc_idx])
pilot_subcar = np.arange(1, N_fft, 2)
active_subcar = subcars[~np.isin(subcars, dead_subcar)]
data_subcar = subcars[~np.isin(subcars, np.concatenate((pilot_subcar, dead_subcar)))]

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

### Extract payload safely
rx_payload = extract_payload(rx, start_index, preamble_len, payload_len)

### Reshape and remove CP
rx_mat = rx_payload.reshape(N_sym, N_fft + N_cp)
rx_no_cp = rx_mat[:, N_cp:]

### FFT -> subcarriers
Y_unshifted = np.fft.fft(rx_no_cp, axis=1)
Y = np.fft.fftshift(Y_unshifted, axes=1)

### Channel estimation from pilots
H_est_full = estimate_channel(Y, pilot_subcar, pilot_symbol, N_fft)

### 1-tap equalization and SER
SER, rx_flat = equalize_and_ser(M, data_subcar, Y, H_est_full, OFDM_CSTL)

print(f"Total symbols:      {len(M[:, data_subcar].flatten())}")
print(f"SER:                {SER:.6f}")

### Plots
fig, axs = plt.subplots(1, 2, figsize=(10, 6))

# |H| over subcarriers
axs[0].plot(range(0, N_fft), np.abs(H_est_full))
axs[0].set_title("|H| over Subcarriers")
axs[0].set_xlabel("Subcarriers")
axs[0].set_ylabel("|H|")
axs[0].grid(True)

# Constellation Plot
axs[1].scatter(OFDM_CSTL.real,
               OFDM_CSTL.imag,
               s=40,
               color="red",
               marker="x",
               label="Ideal")
axs[1].scatter(rx_flat.real, rx_flat.imag, s=4, color="black", label="RX")
axs[1].set_title(f"Equalized Constellation (M = {len(OFDM_CSTL)}, Nc = {len(data_subcar)})")
axs[1].set_xlabel("I")
axs[1].set_ylabel("Q")
axs[1].grid(True)
axs[1].set_aspect('equal', 'box')
axs[1].legend()

plt.tight_layout()
plt.show()
