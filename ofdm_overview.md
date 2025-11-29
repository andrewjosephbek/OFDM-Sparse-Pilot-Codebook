# OFDM Pluto SDR Transceiver - Variable Reference & Proccess Flow

This document explains the variable naming, data structures, and logical DSP flow used in the OFDM transceiver implementation for the Pluto SDR.

The codebase implements a complete OFDM PHY layer, including:

- Preamble based synchronization
- Subcarrier mapping and pilot insertion
- OFDM modulation
- Channel estimation
- 1-tap equalization
- QAM demapping and SER calculation

------------------------------------------------------------------

### Core OFDM Variables

```python
N_fft              # FFT size (number of OFDM subcarriers)
N_cp               # Cyclic prefix length
N_sym              # Number of OFDM symbols in payload
fs                 # Baseband sampling rate
fc                 # RF center frequency
payload_len        # Total time-domain payload length
preamble_len       # Total preamble sample count
full_frame_len     # preamble_len + payload_len
```

------------------------------------------------------------------

### Subcarrier Assignment

```python
dc_idx          # Index of DC subcarrier (center bin)
subcars         # np.arange(N_fft)
dead_subcar     # Subcarriers forced to 0 (DC)
pilot_subcar    # Subcarriers carrying known pilot symbols
data_subcar     # Subcarriers carrying QAM data
active_subcar   # All usable subcarriers except DC
```

------------------------------------------------------------------

### Constellations

```python
QPSK, QAM16    # Modulation alphabets
OFDM_CSTL      # Selected constellation for payload
pilot_symbol   # Known symbol placed on pilot subcarriers
```

All QAM constellations are normalized to unit average power.

------------------------------------------------------------------

### TX Frame Construction

Preamble:
- Random QPSK on active subcarriers
- IFFT to time domain
- CP added
- Repeated twice

Payload:
- M: (N_sym × N_fft) matrix of QAM symbols
- DC zeroed
- Pilots inserted
- IFFT + CP per symbol

Final TX Frame:
```tx_frame = preamble || payload```

------------------------------------------------------------------

### SDR Configuration

```
fs                1 MHz sampling
fc                915 MHz RF
Manual RX gain
Cyclic TX buffer enabled
AGC disabled
RX buffer size set after full_frame_len is known
```

------------------------------------------------------------------

### Synchronization via Correlation

1. Capture samples
2. Correlate with known preamble
3. Find strong peak
4. Compute start_index
5. Extract payload

------------------------------------------------------------------

### OFDM Demodulation Pipeline

1. Reshape rx into (N_sym × (N_fft + N_cp))
2. Remove cyclic prefix
3. FFT each symbol
4. Apply fftshift → subcarrier ordering

------------------------------------------------------------------

### Channel Estimation

```
Y_pilots = Y[:, pilot_subcar]
H_pilots = mean(Y_pilots / pilot_symbol)
```

Interpolation yields H_est_full across all subcarriers.

------------------------------------------------------------------

### Equalization and QAM Detection

Equalization: ```Y_eq = Y / H_est_full```

Energy normalization: ```r = rx_flat / sqrt(mean(|rx_flat|^2))```

Nearest-neighbor QAM detection:
```
d = |r - constellation| 
rx_dec = constellation[argmin(d)]
```

------------------------------------------------------------------

## Process Flow Summary

1. Initialize SDR
2. Generate preamble
3. Generate payload
4. Compute full frame length
5. Set RX buffer size
6. Transmit
7. Capture samples
8. Detect preamble
9. Extract payload
10. Remove CP
11. FFT
12. Channel estimation
13. Equalization
14. QAM demapping
15. SER calculation
16. Plot results

------------------------------------------------------------------

### Notes

- Channel assumed constant over full frame
- No CFO compensation
- Compatible with Pluto (inter-device TX/RX) SDR OTA or loopback use
