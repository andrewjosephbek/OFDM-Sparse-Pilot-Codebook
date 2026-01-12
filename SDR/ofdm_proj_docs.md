# OFDM Pluto SDR Transceiver - Documentation Overview

This script implements a full OFDM transmitter/receiver chain using a Pluto SDR, over the [RemoteRF Platform](https://wireless.ee.ucla.edu/remoterf/).
It includes:

- OFDM frame construction (preamble + payload)
- Subcarrier mapping (data, pilot, dead)
- TX waveform synthesis (IFFT + CP)
- Over-the-air transmission
- Preamble-based synchronization via correlation
- Channel estimation on pilot subcarriers
- 1-tap equalization
- QAM symbol detection and SER computation
- Constellation visualization

---

## Variable Naming Conventions (TX/RX Dimensions)

### Core OFDM Parameters

```
N_fft          Number of subcarriers (FFT size)
N_cp           Cyclic prefix length
N_sym          Number of OFDM symbols in payload
fs             Baseband sampling rate (Hz)
fc             RF center frequency (Hz)
```

### Subcarrier Indexing

```
dc_idx         Index of DC (center) subcarrier
subcars        Array of all subcarrier indices: 0 … N_fft-1
dead_subcar    Subcarriers intentionally forced to zero (e.g., DC)
pilot_subcar   Subcarriers carrying pilot symbols
data_subcar    Subcarriers carrying data symbols
active_subcar  All subcarriers except dead/DC
```

### Constellations

```
QPSK, QAM16    Modulation alphabets
OFDM_CSTL      Selected constellation used for data subcarriers
pilot_symbol   Known constant QPSK pilot symbol
```

### Frame Construction

```
preamble_fd    Frequency-domain preamble
preamble_td    Time-domain preamble (IFFT + CP)
preamble       Final preamble repeated twice
payload_len    Total number of time-domain samples in payload
preamble_len   Number of samples in preamble
full_frame_len Total samples = preamble_len + payload_len
```

### Payload Matrices

```
M              (N_sym × N_fft) matrix of freq-domain OFDM symbols
X              Time-domain payload vector with CPs applied
tx_frame       Final TX frame = preamble || payload
tx_scaled      TX waveform scaled to Pluto DAC range
```

### RX Buffers

```
rx             Raw complex time-domain samples from Pluto
rx_payload     Extracted waveform after preamble alignment
rx_mat         Reshaped into OFDM symbols (rows)
rx_no_cp       CP-removed payload data
```

### FFT Outputs

```
Y_unshifted    FFT outputs (freq bins in natural order)
Y              FFT outputs shifted to subcarrier order (DC in middle)
```

### Channel Estimation

```
Y_pilots       Received pilot subcarriers
H_pilots       Estimated channel on pilot tones
H_est_full     Interpolated full channel estimate (size N_fft)
```

### Equalization and Detection

```
Y_eq           1-tap equalized subcarriers
tx_flat        Transmitted data symbols (flattened)
rx_flat        Equalized received symbols (flattened)
SER            Symbol Error Rate
```

---

## Logical Processing Flow

### 1. Initialization
- Configure Pluto SDR (sample rate, gains)
- Generate subcarrier maps (`data_subcar`, `pilot_subcar`, `dead_subcar`)

### 2. Preamble Construction
- Create randomized QPSK preamble in frequency domain
- Apply IFFT → time domain
- Add cyclic prefix
- Repeat twice for robust detection

### 3. Payload Construction
- Generate random QAM data into matrix `M`
- Insert zeros on dead/DC subcarriers
- Insert known pilot symbols
- For each OFDM symbol:
  - IFFT
  - Add CP
  - Append to payload vector `X`

### 4. Transmission
- Concatenate preamble and payload → `tx_frame`
- Normalize amplitude → `tx_scaled`
- Transmit continuously on Pluto SDR

### 5. Reception
- Flush stale RX samples
- Capture new buffer `rx` from Pluto

### 6. Synchronization / Correlation
- Correlate `rx` with known preamble
- Find strongest peak → estimate frame start
- Extract payload safely (with zero-padding if needed)

### 7. FFT Processing
- Reshape into OFDM symbols
- Remove cyclic prefix
- FFT across symbols
- Apply `fftshift` to align subcarrier ordering

### 8. Channel Estimation
- Extract pilot tones from received subcarriers
- Divide by known pilot symbol
- Average across all symbols
- Interpolate channel estimate across full FFT size

### 9. Equalization + Demodulation
- Apply 1-tap equalizer: ``Y_eq = Y / H_est_full``
- Normalize constellation
- Perform nearest-neighbor QAM detection
- Compute SER

### 10. Visualization
- Plot estimated channel magnitude
- Overlay received constellation vs ideal constellation

---

## Notes

- The channel is assumed quasi-static over the OFDM frame.
- No CFO or STO correction is included (future extension).
- Designed for Pluto SDR in both OTA and loopback setups (inter-device).
