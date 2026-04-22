clear; clc; close all;

% Load raw wav

raw_root = 'raw_waveforms';
out_root = 'denoised_waveforms';

if ~exist(out_root, 'dir')
    mkdir(out_root);
end

groups = dir(raw_root);
groups = groups([groups.isdir] & ~startsWith({groups.name}, '.'));

% Main denoising loop

for g = 1:length(groups)

    group_name = groups(g).name;
    group_in  = fullfile(raw_root, group_name);
    group_out = fullfile(out_root, group_name);

    if ~exist(group_out, 'dir')
        mkdir(group_out);
    end

    files = dir(fullfile(group_in, '*.wav'));

    for f = 1:length(files)

        fname = files(f).name;
        fpath = fullfile(group_in, fname);

        [y, Fs] = audioread(fpath);

        % Handle polarity
        if contains(fname, 'neg')
            y = -y;
        end

        [t, y_denoised] = denoise_AP(y, Fs);

        % Remove '_neg' and '.wav'
        out_name = erase(fname, {'_neg', '.wav'});
        out_file = fullfile(group_out, [out_name '.csv']);

        writematrix([t, y_denoised], out_file);

    end

end

disp('Done')

function [t, y_final] = denoise_AP(y, Fs)

dt = 1/Fs;                         % sampling interval (seconds)
N  = length(y);                    % number of samples

t = (0:N-1).' * dt;                % time vector (column)

% Polynomial baseline subtraction

poly_order = 3; % assume baseline approximated by cubic polynomial

% Identify baseline regions (exclude AP core)

idx = true(N,1); % Creates N x 1 logical vector of true
idx(round(0.25*N):round(0.65*N)) = false; % Sets from 4% to 100% of the window to false where the AP occurs

p = polyfit(t(idx), y(idx), poly_order); % Fits polynomial to the selected baseline points wih poly order of 3
baseline = polyval(p, t); % evaluates the polynomial at all time points

y_detrended = y - baseline; % removes the bias from the full voltage data


% State-space Kalman filter + RTS smoother

% x_k = [voltage; velocity]

A = [1 dt;
     0 1 ];

C = [1 0];

% Process and measurement noise
Q = [1e-6 0;
     0    1e-8];

R = var(y_detrended) * 1e-2;

% Initialize
x_pred = zeros(2, N);
P_pred = zeros(2, 2, N);
x_filt = zeros(2, N);
P_filt = zeros(2, 2, N);

x_filt(:,1) = [y_detrended(1); 0];
P_filt(:,:,1) = eye(2);

% Forward Kalman
for k = 2:N
    % Prediction
    x_pred(:,k) = A * x_filt(:,k-1);
    P_pred(:,:,k) = A * P_filt(:,:,k-1) * A' + Q;

    % Update
    K = P_pred(:,:,k) * C' / (C * P_pred(:,:,k) * C' + R);
    x_filt(:,k) = x_pred(:,k) + K * (y_detrended(k) - C * x_pred(:,k));
    P_filt(:,:,k) = (eye(2) - K * C) * P_pred(:,:,k);
end

% Backwards RTS
x_smooth = x_filt;
P_smooth = P_filt;

for k = N-1:-1:1
    G = P_filt(:,:,k) * A' / P_pred(:,:,k+1);
    x_smooth(:,k) = x_filt(:,k) + ...
        G * (x_smooth(:,k+1) - x_pred(:,k+1));
    P_smooth(:,:,k) = P_filt(:,:,k) + ...
        G * (P_smooth(:,:,k+1) - P_pred(:,:,k+1)) * G';
end

y_kalman = x_smooth(1,:)';

sgolay_order  = 3;
sgolay_window = round(0.05 * N);   % 5% of waveform
sgolay_window = sgolay_window + mod(sgolay_window+1,2);  % force odd

y_final = y_kalman;  % initialize output

y_final(1:N) = ...
    sgolayfilt(y_kalman(1:N), sgolay_order, sgolay_window);

end
