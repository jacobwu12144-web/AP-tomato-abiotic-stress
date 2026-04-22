clc;clearvars;

clear; clc;

root_dir = 'denoised_waveforms';

groups = dir(root_dir);
groups = groups([groups.isdir] & ~startsWith({groups.name}, '.'));

AP_dataset = {};   % dynamic cell array
idx = 1;

for g = 1:length(groups)

    group_name = groups(g).name;        % e.g. 'a_control'
    group_path = fullfile(root_dir, group_name);

    csv_files = dir(fullfile(group_path, '*.csv'));

    for i = 1:length(csv_files)

        disp(length(csv_files))

        file_path = fullfile(group_path, csv_files(i).name);

        data = readmatrix(file_path);
        t = data(:,1);
        V = data(:,2);

        theta = extract_params(t, V);

        % Reject any theta containing NaN or Inf
        if any(~isfinite(theta))
            fprintf('Skipping %s: invalid theta (NaN or Inf)\n', csv_files(i).name);
            continue
        end


        if strcmpi(group_name, 'a_control')
            if isnan(theta(5)) || theta(5) > 20
                fprintf('Skipping %s: tau_rec = %.2f\n', csv_files(i).name, theta(5));
                continue
            end
        end

        % Assign stress label from folder name
        S = group_name;

        AP.theta  = theta;
        AP.stress = S;  
        AP.file   = csv_files(i).name;

        AP_dataset{idx} = AP;
        idx = idx + 1;

    end
end

save('real_AP_dataset.mat', 'AP_dataset');
disp('Dataset construction complete.');

function theta = extract_params(t, V)

theta = nan(1,5);  

N = length(V);

% Baseline
baseline_idx = 1:round(0.1*N);        % first 10% of waveform
Vrest = median(V(baseline_idx));

% Polarity detection
[~, idx_peak] = max(abs(V - Vrest));
polarity = sign(V(idx_peak) - Vrest);
V = polarity * (V - Vrest);

% Amplitude
A = max(V);
if A <= 0
    return
end

% Rise time constant (tau_r)
[~, idx_peak] = max(V);
tpeak = t(idx_peak);

V10 = 0.1 * A;
V90 = 0.9 * A;

% Rising phase
idx = t <= tpeak & V >= V10 & V <= V90;
if sum(idx) < 5
    tau_r = NaN;
else
    y = log(1 - V(idx)/A);
    x = t(idx) - t(1);       % relative to start of recording
    p = polyfit(x, y, 1);
    tau_r = -1 / p(1);
end

% Fall time constant (tau_f)
[~, idx_valley] = min(V);
tvalley = t(idx_valley);

idx = t > tpeak & t <= idx_valley & V >= V10 & V <= V90;
if sum(idx) < 5
    tau_f = NaN;
else
    y = log(V(idx));
    x = t(idx) - tpeak;
    p = polyfit(x, y, 1);
    tau_f = -1 / p(1);
end

% Undershoot amplitude
U = abs(min(V(idx_peak:end)));

% Recovery time constant (tau_rec)
Vrec = V(idx_valley:end);
Vrec = Vrec - min(Vrec);        % shift to positive
Vrec = Vrec / max(Vrec);        % scale 0..1
Vrec = min(max(Vrec, 1e-3), 0.999); % avoid 0 or 1


% log-linear fit
xrec = t(idx_valley:end) - t(idx_valley);
yrec = log(1 - Vrec);
p = polyfit(xrec, yrec, 1);
tau_rec = -1 / p(1);

if tau_rec < 0
    tau_rec = abs(tau_rec);
end

theta = [A, tau_r, tau_f, U, tau_rec];

end

