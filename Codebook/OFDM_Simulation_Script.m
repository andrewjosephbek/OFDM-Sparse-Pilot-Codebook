% OFDM_Simulation_Script.m
clear; close all;

% Monte Carlo Trials
Ntrials = 100;
sweep_pts = 21;

% SYSTEM PARAMETERS
M          = 4;                   % QAM order
Nfft       = 64+1;                 % OFDM size (Carriers, must be odd)
Ncp        = 16;                   % CP length
Nsym       = 1000;                  % Number of OFDM symbols
fs         = 20e6;                  % Sampling frequency (Hz)
pilot_sym  = (1+1j)/sqrt(2);       % Base Pilot Symbol P (Constant Phase for LMMSE)
dead_band_length = 2;

% ADAPTIVE CODEBOOK PARAMETERS
Ncirc      = 1;                    % DISABLED ... Circularly permute pilot position for each row by Ncirc_permute

% CHANNEL PARAMETERS
K_factor_dB = 15;
K_factor   = 10^(K_factor_dB/10);
SNR_dB     = 15;                   % Test SNR in dB
tau_rms    = 10e-9;                % RMS Delay Spread (100 ns) 
fD         = 1;                    % Max Doppler Spread (5 Hz) 

% Codebook Sweep Params
Npf_array = [2, 3, 4, 6, 8];
Npt_array = [100, 150, 200, 250, 300];

tau_array = linspace(1e-9, 200e-9, sweep_pts);
fD_array = linspace(1, 1000, sweep_pts);



%% MONTE CARLO SETUP
NMSE_freq_sweep = zeros(length(Npf_array), length(tau_array));
NMSE_time_sweep = zeros(length(Npt_array), length(fD_array));

BER_freq_sweep = zeros(length(Npf_array), length(tau_array));
BER_time_sweep = zeros(length(Npt_array), length(fD_array));

GP_freq_sweep = zeros(length(Npf_array), length(tau_array));
GP_time_sweep = zeros(length(Npt_array), length(fD_array));

%% A. FIGURE 1: FREQUENCY SELECTIVITY SWEEP (Optimize Nf)
fD_fixed = 1; % Low Doppler
Npt_fixed = 1; % Pilot every symbol

disp('--- Running Frequency Selectivity Sweep (Figure 1) ---');
for i = 1:length(Npf_array) % Loop over pilot spacing curves (Nf)
    Npf_sweep = Npf_array(i);

    for j = 1:length(tau_array) % Loop over x-axis points (tau_rms)
        tau_sweep = tau_array(j);
        [~, AVG_BER, AVG_GP, AVG_H_MSE] = run_OFDM_monte_carlo(Ntrials, M, Nfft, Nsym, Ncp, Npt_fixed, Npf_sweep, pilot_sym, dead_band_length, Ncirc, fs, fD_fixed, tau_sweep, SNR_dB, K_factor);
        NMSE_freq_sweep(i, j) = AVG_H_MSE;
        BER_freq_sweep(i, j) = AVG_BER;
        GP_freq_sweep(i, j) = AVG_GP;

        disp(['  Completed Nf=', num2str(Npf_sweep), ', tau=', num2str(tau_sweep*1e9), ' ns']);
    end
end


%% B. FIGURE 2: TIME VARYING SWEEP (Optimize Nt)
tau_fixed = 10e-9; % Low Delay Spread
Npf_fixed = 2;    % Dense frequency pilot

disp('--- Running Time Variation Sweep (Figure 2) ---');
for i = 1:length(Npt_array) % Loop over time spacing curves (Nt)
    Npt_sweep = Npt_array(i);

    for j = 1:length(fD_array) % Loop over x-axis points (fD)
        fD_sweep = fD_array(j);

        % Accumulate NMSE over Ntrials
        [~, AVG_BER, AVG_GP, AVG_H_MSE] = run_OFDM_monte_carlo(Ntrials, M, Nfft, Nsym, Ncp, Npt_sweep, Npf_fixed, pilot_sym, dead_band_length, Ncirc, fs, fD_sweep, tau_fixed, SNR_dB, K_factor);

        NMSE_time_sweep(i, j) = AVG_H_MSE;
        BER_time_sweep(i, j) = AVG_BER;
        GP_time_sweep(i, j) = AVG_GP;

        disp(['  Completed Nt=', num2str(Npt_sweep), ', fD=', num2str(fD_sweep), ' Hz']);
    end
end

%% Gain Over Static (Freq Selective) Estimation
thresh_f = 0.06;
gp_adaptive_f = zeros(size(tau_array));

disp('Freq Sweep Codebook:  ')

for i = 1:length(tau_array)
   A = NMSE_freq_sweep(:, i);
   B = GP_freq_sweep(:, i);
   thresh_mask = A < thresh_f;
   good_idxs =  find(thresh_mask);

   best_idx = max(good_idxs);
   best_spacing = Npf_array(best_idx);

   if(sum(thresh_mask) == 0)
        gp_adaptive_f(i) = B(1);
        disp(['tau RMS: ', num2str(tau_array(i)*10^9), ' ns   Nf = min'])

   else
        gp_adaptive_f(i) = B(best_idx);
        disp(['tau RMS: ', num2str(tau_array(i)*10^9), ' ns   Nf = ', num2str(best_spacing)])
   end
end

%Gain Over Static (Time Selective) Estimation
thresh_t = 0.06;
gp_adaptive_t = zeros(size(fD_array));

disp('Time Sweep Codebook:  ')

for i = 1:length(fD_array)
   A = NMSE_time_sweep(:, i);
   B = GP_time_sweep(:, i);
   thresh_mask = A < thresh_f;
   good_idxs =  find(thresh_mask);
   
   best_idx = max(good_idxs);
   best_spacing = Npt_array(best_idx);
   
   if(sum(thresh_mask) == 0)
        gp_adaptive_t(i) = B(1);
        disp(['fD: ', num2str(fD_array(i)), ' Hz   Nt = min'])
   else
        gp_adaptive_t(i) = B(best_idx);
        disp(['fD: ', num2str(fD_array(i)), ' Hz   Nt = ', num2str(best_spacing)])
   end
end

avg_gp_gain_f = mean(gp_adaptive_f - GP_freq_sweep(1, :));
avg_gp_f = mean(GP_freq_sweep(1, :));

gp_gain_pct_f = (avg_gp_gain_f / avg_gp_f) * 100;

avg_gp_gain_t = mean(gp_adaptive_t - GP_time_sweep(1, :));
avg_gp_t = mean(GP_time_sweep(1, :));

gp_gain_pct_t = (avg_gp_gain_t / avg_gp_t) * 100;

disp(['  Mean Goodputgain over delay sweep:  ', num2str(gp_gain_pct_f), '%']);
disp(['  Mean Goodputgain over doppler sweep:', num2str(gp_gain_pct_t), '%']);


%% PLOT GENERATION

set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');
set(groot, 'defaultColorbarFontName', 'Times New Roman');


% Convert tau_array to nanoseconds for clean plotting
tau_ns = tau_array * 1e9;

figure(1);

hold on;
for i = 1:length(Npf_array)
    plot(tau_ns, NMSE_freq_sweep(i, :), 'DisplayName', ['N_f = ', num2str(Npf_array(i))]);
end
title('NMSE vs. Delay Spread (Frequency Selectivity)');
xlabel('RMS Delay Spread \tau_{rms} (ns)');
ylabel('NMSE (Normalized Mean Square Error)');
% set(gca, 'YScale', 'log');
grid on;
legend('show', 'Location', 'NorthWest');

figure(2);

hold on;
for i = 1:length(Npt_array)
    plot(fD_array, NMSE_time_sweep(i, :), 'DisplayName', ['N_t = ', num2str(Npt_array(i))]);
end
title('NMSE vs. Doppler Spread (Time Selectivity)');
xlabel('Maximum Doppler Spread f_D (Hz)');
ylabel('NMSE (Normalized Mean Square Error)');
% set(gca, 'YScale', 'log');
grid on;
legend('show', 'Location', 'NorthWest');

figure(3);

hold on;
for i = 1:length(Npf_array)
    plot(tau_ns, BER_freq_sweep(i, :), 'DisplayName', ['N_f = ', num2str(Npf_array(i))]);
end
title('BER vs. Delay Spread (Frequency Selectivity)');
xlabel('RMS Delay Spread \tau_{rms} (ns)');
ylabel('BER (Bit Error)');
% set(gca, 'YScale', 'log');
grid on;
legend('show', 'Location', 'NorthWest');

figure(4);

hold on;
for i = 1:length(Npt_array)
    plot(fD_array, BER_time_sweep(i, :), 'DisplayName', ['N_t = ', num2str(Npt_array(i))]);
end
title('BER vs. Doppler Spread (Time Selectivity)');
xlabel('Maximum Doppler Spread f_D (Hz)');
ylabel('BER (Bit Error)');
% set(gca, 'YScale', 'log');
grid on;
legend('show', 'Location', 'NorthWest');

figure(5);

hold on;
for i = 1:length(Npf_array)
    plot(tau_ns, GP_freq_sweep(i, :), 'DisplayName', ['N_f = ', num2str(Npf_array(i))]);
end
title('Goodput vs. Delay Spread (Frequency Selectivity)');
xlabel('RMS Delay Spread \tau_{rms} (ns)');
ylabel('Goodput (Bits/symbol)');
% set(gca, 'YScale', 'log');
grid on;
legend('show', 'Location', 'NorthEast');

figure(6);

hold on;
for i = 1:length(Npt_array)
    plot(fD_array, GP_time_sweep(i, :), 'DisplayName', ['N_t = ', num2str(Npt_array(i))]);
end
title('Goodput vs. Doppler Spread (Time Selectivity)');
xlabel('Maximum Doppler Spread f_D (Hz)');
ylabel('Goodput (Bits/symbol)');
% set(gca, 'YScale', 'log');
grid on;
legend('show', 'Location', 'NorthEast');

figure(7);

hold on;
plot(fD_array, gp_adaptive_f, 'DisplayName', ['N_f = adaptive']);
plot(fD_array, GP_freq_sweep(1, :), 'DisplayName', ['N_f = 2']);

title('Adaptive Goodput (Frequency Selectivity Sweep)');
xlabel('RMS Delay Spread \tau_{rms} (ns)');
ylabel('Goodput (Bits/symbol)');
% set(gca, 'YScale', 'log');
grid on;
legend('show', 'Location', 'NorthEast');


figure(8);

hold on;
plot(fD_array, gp_adaptive_t, 'DisplayName', ['N_t = adaptive']);
plot(fD_array, GP_time_sweep(1, :), 'DisplayName', ['N_t = 100']);
title('Adaptive Goodput (Time Selectivity Sweep)');
xlabel('Maximum Doppler Spread f_D (Hz)');
ylabel('Goodput (Bits/symbol)');
% set(gca, 'YScale', 'log');
grid on;
legend('show', 'Location', 'NorthEast');

FolderName = "C:\Users\andre\Documents\GitHub\239-Project\figs";

%% Save figures 1 through 8
for figNum = 1:8
    figure(figNum); % bring figure to focus (optional but safe)
    fig = gcf;

    fig.Units = 'inches';
    fig.Position = [1 1 6 4];   % change this to your preferred size

    lines = findall(fig, 'Type', 'Line');
    set(lines, 'LineWidth', 1.5);
    filename = fullfile(FolderName, sprintf('Figure%d.png', figNum));

    % Save as PNG at high resolution
    saveas(gcf, filename);
    % Alternatively (higher quality):
    % exportgraphics(gcf, filename, 'Resolution', 300);
end

disp('All figures saved successfully.');
