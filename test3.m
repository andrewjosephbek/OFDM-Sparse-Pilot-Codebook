
clear; clc; close all;
rng(1);   % reproducible

%  System Parameters
fc         = 100e9;
c          = 3e8;
velocities = linspace(0, 50, 21);  % 0–50 m/s
M          = 16;                   % QAM order
N          = 64;                   % OFDM size
Ncp        = 16;                   % CP length
numSymbols = 2000;
fs         = 1e6;
Ts         = 1/fs;
SNR_dB     = 200;

% Choose which subcarriers carry data (1..N)
data_carriers = 31+[-26:-22 -20:-8 -6:-1 1:6 8:20 22:26];
N_data        = numel(data_carriers);

pilot_carriers = 31+[-21 -7 7 21];  


% Channel Params
num_paths      = 2;
max_delay_samp = 3; % < Ncp

% Fixed channel geometry
[delays, base_gains, doppler_norm] = init_channel_geometry(num_paths, max_delay_samp);

BER = zeros(size(velocities));

%  Sweep velocity 
for i = 1:numel(velocities)

    v = velocities(i);
    max_doppler = (v / c) * fc;     % Hz
    fd_vec = doppler_norm * max_doppler;

    BER(i) = run_ofdm_qam_frame_BER( ...
        M, N, Ncp, numSymbols, fs, data_carriers, pilot_carriers, SNR_dB, ...
        delays, base_gains, fd_vec);

    fprintf("v = %5.1f m/s  → f_D,max = %8.1f Hz,   BER = %.4f\n", ...
             v, max_doppler, BER(i));
end

% Plot BER vs velocity
figure;
semilogy(velocities, BER, 'o-', 'LineWidth', 2);
xlabel('Velocity (m/s)');
ylabel('BER');
title(sprintf('OFDM, %d-QAM BER vs Velocity (mmWave fc=%.1f GHz)', ...
       M, fc/1e9));
grid on;


%  OFDM Frame with M-QAM
function BER = run_ofdm_qam_frame_BER( ...
    M, N, Ncp, numSymbols, fs, data_carriers, pilot_carriers, SNR_dB, ...
    delays, base_gains, fd_vec)

    Ts = 1/fs;
    L = numel(delays);
    N_data = numel(data_carriers);

    % Only generate symbols for data subcarriers
    numSymsTot = N_data * numSymbols;
    bitsPerSym = log2(M);
    numBitsTot = numSymsTot * bitsPerSym;

    % Generate random bits
    tx_bits = randi([0 1], numBitsTot, 1);
    tx_idx  = bi2de(reshape(tx_bits, bitsPerSym, []).', 'left-msb');
    tx_sym  = qammod(tx_idx, M, 'UnitAveragePower', true);   % col vec

    % Arrange into [N_data x numSymbols]
    X_data = reshape(tx_sym, N_data, numSymbols);

    % Full OFDM grid
    X = zeros(N, numSymbols);
    X(data_carriers, :) = X_data;
    X(pilot_carriers, :) = 1;

    % OFDM IFFT + CP
    tx = zeros((N+Ncp)*numSymbols, 1);
    for m = 1:numSymbols
        x_td = ifft(X(:,m), N);
        ind = (m-1)*(N+Ncp)+1 : m*(N+Ncp);
        tx(ind) = [x_td(end-Ncp+1:end); x_td];
    end

    numSamples = length(tx);

    % Time varying channel paths
    n = (0:numSamples-1);
    t = n * Ts;
    h_paths = zeros(L, numSamples);
    for p = 1:L
        h_paths(p,:) = base_gains(p) * exp(1j*2*pi*fd_vec(p)*t);
    end

    % Apply channel
    rx = zeros(size(tx));
    for p = 1:L
        d  = delays(p);
        hp = h_paths(p,:).';
        rx(1+d:end) = rx(1+d:end) + hp(1:end-d) .* tx(1:end-d);
    end

    % AWGN
    sig_pow = mean(abs(rx).^2);
    SNR     = 10^(SNR_dB/10);
    noise   = sqrt(sig_pow/(2*SNR))*(randn(size(rx))+1j*randn(size(rx)));
    rx      = rx + noise;

    % Receiver FFT
    rx_blocks = reshape(rx, N+Ncp, numSymbols);
    Y = zeros(N, numSymbols);
    for m = 1:numSymbols
        r_blk  = rx_blocks(:,m);
        r_data = r_blk(Ncp+1:end);
        Y(:,m) = fft(r_data, N);
    end

    % Pilot based channel estimation (per symbol, frequency interpolation)
    Hhat_data = zeros(N_data, numSymbols);

    for m = 1:numSymbols
        % 4×1 received pilots for symbol m
        Y_pilots = Y(pilot_carriers, m);

        % 4×1 transmitted pilots for symbol m (all ones here)
        X_pilots = X(pilot_carriers, m);

        % LS estimate on pilot tones: H_pilots = Y / X
        H_pilots = Y_pilots ./ X_pilots;   % 4×1

        % Interpolate across data subcarriers
        Hhat_data(:, m) = interp1( ...
            pilot_carriers, ...   % x: pilot subcarrier indices (1..N)
            H_pilots, ...         % y: channel at pilots
            data_carriers, ...    % xq: data subcarrier indices (1..N)
            'linear', 'extrap');  % N_data×1
    end

    % Equalize data subcarriers
    Y_data = Y(data_carriers, :);              % N_data x numSymbols
    Z_data = Y_data ./ (Hhat_data + 1e-12);    % one-tap ZF EQ
    z_vec  = Z_data(:);                        % vectorize

    % Demodulate QAM, get BER
    rx_idx  = qamdemod(z_vec, M, 'UnitAveragePower', true);
    rx_bits = de2bi(rx_idx, bitsPerSym, 'left-msb').';
    rx_bits = rx_bits(:);

    Nbits = min(length(rx_bits), length(tx_bits));
    BER   = mean(rx_bits(1:Nbits) ~= tx_bits(1:Nbits));
end




%  Channel geometry fixed across velocities
function [delays, base_gains, doppler_norm] = init_channel_geometry( ...
    L, max_delay_samp)

    delays = randi([0 max_delay_samp], L, 1);

    pdp = exp(-(0:L-1).' / max(1, L/3));
    pdp = pdp / sum(pdp);

    g0  = (randn(L,1)+1j*randn(L,1))/sqrt(2);
    base_gains = g0 .* sqrt(pdp);

    doppler_norm = 2*rand(L,1) - 1;   % range [-1,1]
end


%  Per-subcarrier channel for OFDM
function H = estimate_H_ofdm(N, Ncp, numSymbols, h_paths, delays, numSamples)

    L = numel(delays);
    H = zeros(N, numSymbols);

    for m = 1:numSymbols
        n0 = (m-1)*(N+Ncp) + Ncp + 1;
        n0 = min(n0, numSamples);

        a = h_paths(:, n0);     % path gains at start of data
        k = (0:N-1).';

        H_k = zeros(N,1);
        for p = 1:L
            H_k = H_k + a(p) * exp(-1j*2*pi*k*(delays(p)/N));
        end
        H(:,m) = H_k;
    end
end
