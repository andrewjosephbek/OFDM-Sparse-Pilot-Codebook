# Sparse-Pilot OFDM Transceiver: Adaptive Codebook Design

This repository contains the implementation of an adaptive 2D pilot codebook for OFDM systems. The system is designed to optimize spectral efficiency and goodput by dynamically selecting pilot patterns based on real-time channel conditions, specifically tracking delay spread and Doppler spread.

## Project Overview

Reliable communication in OFDM systems requires accurate Channel Frequency Response (CFR) estimation. Traditional fixed pilot patterns are often over-designed to handle worst-case fading, which wastes bandwidth. This project introduces a lookup table-based approach that selects the minimum required pilot density to maintain a target Bit Error Rate (BER).

### Key Achievements
* **Goodput Optimization:** Achieved a 38.49% increase in goodput over delay sweeps and a 0.143% increase over Doppler sweeps compared to non-adaptive baselines.
* **Efficient Tracking:** Successfully maps channel parameters (Delay/Doppler) to optimal pilot spacing ($N_f \times N_t$).
* **Reliability:** Maintained a Normalized Mean Squared Error (NMSE) below 6% across all adaptive transitions.

## Technical Specifications

| Parameter | Configuration |
| :--- | :--- |
| **Waveform** | OFDM |
| **Modulation** | QPSK / 16-QAM |
| **FFT Size ($N_{fft}$)** | 64 |
| **Cyclic Prefix ($N_{cp}$)** | 16 |
| **Sampling Frequency** | 20 MHz |
| **SDR Hardware** | ADALM-Pluto (for over-the-air validation) |

## System Architecture

The project characterizes channel distortion through two primary phenomena:
1. **Frequency Selectivity:** Caused by Delay Spread ($\tau_{rms}$), limiting Coherence Bandwidth.
2. **Time Selectivity:** Caused by Doppler Spread ($f_D$), limiting Coherence Time.



The transceiver uses a **Least Squares (LS) Estimator** combined with **2D Linear Interpolation**. This low-complexity approach allows the receiver to estimate the full channel grid from sparse pilot symbols before performing one-tap equalization.

## Adaptive Codebook Lookup Table

The following boundaries define the maximum pilot sparsity permissible while maintaining the target 6% NMSE (BER $\sim5\times10^{-3}$).

| $N_f$ (Freq Spacing) | $\tau_{rms}$ Range | $N_t$ (Time Spacing) | $f_D$ Range |
| :--- | :--- | :--- | :--- |
| 8 | 0 ns – 50 ns | 300 | 0 Hz – 150 Hz |
| 6 | 50 ns – 60 ns | 250 | 150 Hz – 300 Hz |
| 4 | 60 ns – 100 ns | 200 | 300 Hz – 400 Hz |
| 3 | 100 ns – 140 ns | 150 | 400 Hz – 600 Hz |
| 2 | 140 ns – 200 ns | 100 | 600 Hz – 1000 Hz |

## Results & Validation

The system was validated using a Matlab Monte Carlo analysis (1000 trials). 
* **Frequency Sweep:** $N_f$ was swept against increasing RMS Delay Spread (1–200 ns).
* **Time Sweep:** $N_t$ was swept against increasing Maximum Doppler Spread (1–1000 Hz).

The resulting "stair-step" goodput curves demonstrate the discrete switching between pilot patterns, ensuring that the system only uses the bandwidth necessary for the current environment.
