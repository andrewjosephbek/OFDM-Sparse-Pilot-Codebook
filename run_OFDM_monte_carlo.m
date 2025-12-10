% run_OFDM_monte_carlo.m

function [AVG_SER, AVG_BER, AVG_Goodput, AVG_H_MSE] = run_OFDM_monte_carlo(Ntrials, M, Nfft, Nsym, Ncp, Npt, Npf, pilot_sym, dead_band_length, Ncirc, fs, fD, tau_rms, SNR_dB, K_factor)
    dead_lower = dead_band_length;
    dead_upper = Nfft-dead_band_length;

    dead_carriers = [1:dead_lower, ceil(Nfft/2), dead_upper:Nfft];
    pilot_carriers = generate_pilots(dead_lower+1, dead_upper-1, Npf); 
    all_carriers = 1:Nfft;
    live_carriers = array_set_difference(all_carriers, dead_carriers);

    AVG_SER = 0;
    AVG_BER = 0;
    AVG_Goodput = 0;
    AVG_H_ERROR = 0;
    AVG_H_MAG = 0;

    for n = 1:Ntrials
    
        % 3. TRANSMITTER (TX)
        [tx, ~, Xf, X_data_mask, X_pilot_mask] = build_OFDM_tx(Nfft, Nsym, Ncp, M, pilot_carriers, dead_carriers, pilot_sym, Npt, Ncirc);
        
        % 4. CHANNEL MODEL
        rx_serial = apply_fading_channel(tx, fs, fD, tau_rms, SNR_dB, 1, K_factor);
    
        % 5. RECEIVER CORE (RX Core)
        Yf = build_OFDM_rx_core(rx_serial, Nfft, Ncp, Nsym);
    
        % 6. EQUALIZATION (LMMSE)
        [Y_eq, Hf_est] = equalize_ofdm_data(Yf, X_data_mask, X_pilot_mask, live_carriers, pilot_sym);
        [Hf_true] = genie_channel(Yf, Xf, live_carriers);
    
        % 7. DEMODULATION AND SER CALCULATION
        [SER, BER, Goodput_bps, H_error, H_mag] = calculate_performance_metrics(Y_eq, Xf, X_data_mask, Hf_true, Hf_est, tx, M);
    
        AVG_SER = AVG_SER + SER;
        AVG_BER = AVG_BER + BER;
        AVG_Goodput = AVG_Goodput + Goodput_bps;
        AVG_H_ERROR = AVG_H_ERROR + H_error;
        AVG_H_MAG = AVG_H_MAG + H_mag;
    end

    AVG_SER = AVG_SER / Ntrials;
    AVG_BER = AVG_BER  / Ntrials;
    AVG_Goodput = AVG_Goodput / Ntrials;
    AVG_H_ERROR = AVG_H_ERROR / Ntrials;
    AVG_H_MAG = AVG_H_MAG / Ntrials;
    
    AVG_H_MSE = AVG_H_ERROR / AVG_H_MAG;
end


function [pilots] = generate_pilots(K, L, density)
% ... (generate_pilots function remains unchanged) ...
    
    % Calculate the total range length
    Range_Length = L - K;
    N = ceil(Range_Length / density) + 1;
    non_integer_pilots = linspace(K, L, N);
    
    center_value = (K + L) / 2;
    tolerance = 1e-9;
    is_center = abs(non_integer_pilots - center_value) < tolerance;
    non_integer_pilots(is_center) = non_integer_pilots(is_center) + 1;
    
    pilots = round(non_integer_pilots);
    pilots = unique(pilots);
    
    K_rounded = round(K);
    L_rounded = round(L);
    
    if pilots(1) ~= K_rounded
        pilots = [K_rounded, pilots];
    end
    if pilots(end) ~= L_rounded
        pilots = [pilots, L_rounded];
    end
    
    pilots = reshape(pilots, 1, []);
end

function [tx, Xt, Xf, X_data_mask, X_pilot_mask] = build_OFDM_tx(Nfft, Nsym, Ncp, M, pilot_carriers, dead_carriers, pilot_sym, Npt, Ncirc)
    
    % X is now (Nfft x Nsym) -> (Carriers x Symbols)
    X = zeros(Nfft, Nsym); 

    % --- 1. PILOT GENERATION AND INSERTION ---
    
    X_pilot_mask = false(size(X));

    for m = 1:Nsym
        if(mod(m, Npt) == 0)
                X(pilot_carriers, m) = pilot_sym;
                X_pilot_mask(pilot_carriers, m) = 1;
            % if(Ncirc ~= 0 &&mod(m, 2) == 1)
            %     X(:, m) = circshift(X(:, m), Ncirc);
            %     X_pilot_mask(:, m) = circshift(X_pilot_mask(:, m), Ncirc);
            % end
        end
    end
    
    % --- 2. DATA MASK AND SYMBOL GENERATION ---
    
    % Create the DATA mask (X_data_mask) - Size is Nfft x Nsym
    X_data_mask = true(Nfft, Nsym);
    
    % Rule A: Exclude all dead carriers (ROWS/Carriers).
    X_data_mask(dead_carriers, :) = false;
    
    % Rule B: Exclude pilot carriers ONLY where pilots are ON (using the pilot mask).
    X_data_mask(X_pilot_mask) = false;
    
    % --- 3. DATA GENERATION AND INSERTION ---
    
    num_data_symbols_needed = sum(X_data_mask(:));
    bitsPerSym = log2(M);
    numBitsTot = num_data_symbols_needed * bitsPerSym;
    
    % Generate random bits and QAM map, sized exactly to fill the mask.
    tx_bits = randi([0 1], numBitsTot, 1);
    tx_idx  = bi2de(reshape(tx_bits, bitsPerSym, []).', 'left-msb');
    tx_sym  = qammod(tx_idx, M, 'UnitAveragePower', true); 
    
    % Assign the flattened stream of data symbols into the TRUE positions of X.
    X(X_data_mask) = tx_sym;

    % --- 4. FINAL CLEAN UP AND TRANSMISSION ---
    
    % Clear dead carriers (set ROWS to zero) - Redundant but good check
    X(dead_carriers, :) = 0;
    Xf = X;

    % IFFT along dimension 1 (ROWS/Carriers)
    Xt = ifft(X, Nfft, 1); 
    
    % CP is the last Ncp ROWS
    cp_rows = (Nfft - Ncp + 1):Nfft;
    
    % Concatenate CP (Vertical Concatenation)
    Xt_cp = [Xt(cp_rows, :); Xt]; 
    
    % Flatten (Column-Major order, which is Symbol-by-Symbol)
    tx = reshape(Xt_cp, 1, numel(Xt_cp));
    
end

function [Yf] = build_OFDM_rx_core(rx_serial, Nfft, Ncp, Nsym)

    % --- CP Removal and FFT ---
    samples_per_symbol = Nfft + Ncp;
    
    % 1. Reshape: Flattened signal to (Nfft+Ncp) rows x Nsym columns (Reverse of Tx)
    Y_cp = reshape(rx_serial, samples_per_symbol, Nsym); 
    
    % 2. CP Removal: Remove the first Ncp ROWS
    Y_no_cp = Y_cp(Ncp + 1 : end, :);
    
    % 3. FFT: Along Dimension 1 (ROWS/Carriers)
    Yf = fft(Y_no_cp, Nfft, 1); 
    
end

function [Y_eq, H_eq] = equalize_ofdm_data(Yf, X_data_mask, X_pilot_mask, live_carriers, pilot_sym)

    X_pilots_tx_reconstructed = pilot_sym*X_pilot_mask;
    
    % New equalization
    H_eq = (Yf.*X_pilot_mask) ./ X_pilots_tx_reconstructed;

    H_eq (H_eq ==0) = NaN;
    H_eq = fillmissing(H_eq, 'linear', 1);
    H_eq = fillmissing(H_eq, 'linear', 2);

    Y_symbol_equalized = Yf ./ H_eq;

    Y_eq = Y_symbol_equalized(X_data_mask);
    H_eq = H_eq(live_carriers, :);
end

function [Htrue] = genie_channel(Yf, X, live_carriers)
    Htrue = Yf(live_carriers, :) ./ X(live_carriers, :);
end

function rx_serial = apply_fading_channel(tx_cp_serial, fs, fD, tau_rms, SNR_dB, use_AWGN, K_factor)

    Ts = 1 / fs; 
    T_max = 5 * tau_rms; 
    
    % --- 1. Define the Channel Taps (PDP for Scattered Components) ---
    delays = 0 : Ts : T_max;
    if length(delays) > 20
        delays = delays(1:20); 
    end
    
    gains_linear = exp(-delays / tau_rms);
    
    % Normalize gains so total power of ALL taps is 1 (0 dB) before K-factor weighting
    average_path_gains_dB = 10*log10(gains_linear); 
    
    % --- 2. Configure the Rician Channel Object ---
    
    channel = comm.RicianChannel;
    
    channel.SampleRate = fs;
    channel.PathDelays = delays;
    channel.AveragePathGains = average_path_gains_dB;
    channel.KFactor = K_factor; 
    channel.DopplerSpectrum = doppler("Jakes");
    channel.DirectPathDopplerShift = 0; 
    channel.DirectPathInitialPhase = 0.5; 
    channel.MaximumDopplerShift = fD;
    channel.Visualization = 'Off';
    
    % --- 3. Apply the Channel Fading ---
    tx_cp_col = tx_cp_serial.'; 
    faded_signal = channel(tx_cp_col);
    
    % --- 4. Add Additive White Gaussian Noise (AWGN) ---
    if(use_AWGN)
        rx_serial_col = awgn(faded_signal, SNR_dB, 'measured');
        rx_serial = rx_serial_col.'; 
    else
        rx_serial = faded_signal.';
    end
    
end


function [BER] = calculate_BER(Y_eq, tx_sym_original, M)
% CALCULATE_BER computes the Bit Error Rate (BER) by comparing transmitted and received bits.
%   Y_eq: Stream of equalized received symbols.
%   tx_sym_original: Stream of original transmitted data symbols.
%   M: QAM order (e.g., 16).

    bits_per_symbol = log2(M);
    
    % 1. Symbol De-mapping (Nearest Neighbor Estimation)
    rx_idx = qamdemod(Y_eq, M, 'UnitAveragePower', true);
    tx_idx_original = qamdemod(tx_sym_original, M, 'UnitAveragePower', true);
    
    % 2. Convert Indices to Bits
    tx_bits_original = de2bi(tx_idx_original, bits_per_symbol, 'left-msb');
    tx_bits_original = tx_bits_original(:);
    rx_bits_recovered = de2bi(rx_idx, bits_per_symbol, 'left-msb');
    rx_bits_recovered = rx_bits_recovered(:);
    
    % 3. Calculate BER
    number_of_bit_errors = sum(tx_bits_original ~= rx_bits_recovered);
    total_transmitted_bits = length(tx_bits_original);
    
    BER = number_of_bit_errors / total_transmitted_bits;
end

function [Goodput_bits_per_sample, total_samples_sent] = calculate_Goodput(tx, BER, M, X_data_mask)
% CALCULATE_GOODPUT computes the effective Goodput in bits per total complex sample.
%   tx: The full serial transmitted signal (used to find total samples sent).
%   BER: The calculated Bit Error Rate.
%   M: QAM order.

    bits_per_symbol = log2(M);
    
    total_samples_sent = length(tx);
    
    total_data_sym = nnz(X_data_mask);
    total_bits_sent = total_data_sym * bits_per_symbol;
    
    % Correctly received bits = Total bits sent * (1 - BER)
    correctly_received_bits = total_bits_sent * (1 - BER);
    
    Goodput_bits_per_sample = correctly_received_bits / total_samples_sent;
end

function array_diff = array_set_difference(array1, array2)
    is_present = ismember(array1, array2);

    is_absent = ~is_present;

    array_diff = array1(is_absent);
end


function [SER, BER, Goodput_bps, H_error, H_mag] = calculate_performance_metrics(Y_eq, Xf, X_data_mask, Hf_true, Hf_est, tx, M)

    % --- 1. Symbol Error Rate (SER) Calculation ---
    
    tx_sym_original = Xf(X_data_mask);
    % De-map equalized symbols and original symbols to indices
    rx_idx = qamdemod(Y_eq, M, 'UnitAveragePower', true);
    tx_idx_original = qamdemod(tx_sym_original, M, 'UnitAveragePower', true);
    
    % Calculate symbol errors
    number_of_symbol_errors = sum(tx_idx_original ~= rx_idx);
    total_data_symbols = length(tx_sym_original);
    SER = number_of_symbol_errors / total_data_symbols;
    
    % --- 2. Bit Error Rate (BER) and Goodput ---
    
    % Assumes calculate_BER and calculate_Goodput helpers are defined elsewhere.
    BER = calculate_BER(Y_eq, tx_sym_original, M);
    [Goodput_bps, ~] = calculate_Goodput(tx, BER, M, X_data_mask);

    % --- 3. Channel Estimation Error Metrics (NMSE Components) ---
    
    % Calculate the Total Squared Error (Error Power, Numerator for NMSE)
    % This measures the magnitude of the difference between the true and estimated channel.
    H_error = mean(abs(Hf_true - Hf_est).^2, 'all');
    
    % Calculate the Total Channel Power (Denominator for NMSE)
    % This measures the magnitude of the true channel response.
    H_mag = mean(abs(Hf_true).^2, 'all');

end