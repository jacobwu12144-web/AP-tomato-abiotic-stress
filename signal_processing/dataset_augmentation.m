
clc; close all;

load('real_AP_dataset.mat');

% Extract stress labels correctly
stress_labels = cellfun(@(x) x.stress, AP_dataset, 'UniformOutput', false);
unique_stress = unique(stress_labels);

sorted_stress = cell(length(unique_stress), 1);

for i = 1:length(unique_stress)
    idx = strcmp(stress_labels, unique_stress{i});
    sorted_stress{i} = AP_dataset(idx);
end


% Extract theta matrices for each stressor

Theta_by_stress = cell(length(unique_stress), 1); % Each spot in the cell array stores a full matrix for a given stressor

for i = 1:length(unique_stress)
    APs = sorted_stress{i};
    N = length(APs);

    Theta = zeros(N, length(APs{1}.theta)); % preallocates a matrix for all the APs of a given stress

    for j = 1:N
        Theta(j, :) = APs{j}.theta; % we assemble the matrix row by row
    end

    Theta_by_stress{i} = Theta; % store matrix back into the cell array
end

% Distribution overlays by parameter and stress

param_names = {'Amplitude A', 'Rise Time \tau_r', 'Fall Time \tau_f', ...
               'Undershoot U', 'Recovery Time \tau_{rec}'};

num_params = 5;
num_stress = length(unique_stress);

colors = lines(num_stress);

figure('Color','w','Position',[100 100 1400 300]);

for p = 1:num_params
    subplot(1, num_params, p); hold on;
    
    for s = 1:num_stress
        Theta = Theta_by_stress{s};
        
        % Log-transform (matches your modeling assumptions)
        epsilon = 1e-6;  % tiny positive value
        vals = Theta(:,p);
        
        % Sanity check: see if any <= 0
        if any(vals <= 0)
            disp(['Warning: Theta values <=0 for parameter ', param_names{p}]);
            disp(vals(vals <= 0))
        end
        
        % Clamp to avoid log of zero or negative
        vals = max(vals, epsilon);
        
        % Log-transform
        x = log(vals);

        
        % Kernel density estimate
        [f, xi] = ksdensity(x);
        
        plot(xi, f, ...
            'LineWidth', 2, ...
            'Color', colors(s,:));
    end
    
    title(param_names{p}, 'Interpreter','tex')
    xlabel('log(parameter value)')
    ylabel('Density')
    grid on
    
    if p == num_params
        legend(unique_stress, 'Location','eastoutside')
    end
end

sgtitle('Empirical Parameter Distributions by Stress Condition (Log Space)');



% Learn distributions for each stressor

mu_stress = cell(length(unique_stress), 1); % Preallocate a cell array for all mu values
Sigma_stress = cell(length(unique_stress), 1); % Prealloate a cell array for all sigma values
Nk = zeros(length(unique_stress), 1); % This is the empirical mixing weight operator

for i = 1:length(unique_stress)
    Theta = Theta_by_stress{i};
    
    % Log-transform positive parameters:
    Theta_log = log(Theta);
    
    % Compute mean and regularized covariance
    mu = mean(Theta_log,1);
    Sigma = cov(Theta_log);
    
    % Regularization: add small diagonal to prevent singular covariance
    Sigma = Sigma + 1e-6 * eye(size(Sigma));
    
    mu_stress{i} = mu;
    Sigma_stress{i} = Sigma;
end

% Assign number of synthetic per group

M_by_stress = zeros(length(unique_stress),1);

for i = 1:length(unique_stress)
    if strcmp(unique_stress{i}, 'a_control')
        M_by_stress(i) = 3000;  % control
    else
        M_by_stress(i) = 1000;  % each stressed category
    end
end

% Sample synthetic parameters in log-space
Theta_syn_by_stress = cell(length(unique_stress),1);

for i = 1:length(unique_stress)
    mu = mu_stress{i};
    Sigma = Sigma_stress{i};
    M = M_by_stress(i);  % get category-specific sample size
    
    % Sample log-space parameters
    Theta_log_syn = mvnrnd(mu, Sigma, M);
    
    % Back-transform exponential for positive parameters
    Theta_syn = exp(Theta_log_syn);
    
    Theta_syn_by_stress{i} = Theta_syn;
end

%% 5. Assign binary labels

% Real dataset labels (unchanged)
binary_labels_real = zeros(length(AP_dataset),1);
for i = 1:length(AP_dataset)
    if ~strcmp(AP_dataset{i}.stress,'a_control')
        binary_labels_real(i) = 1;
    end
end

% Synthetic dataset labels
Theta_syn_all = vertcat(Theta_syn_by_stress{:});  % stack all synthetic APs
binary_labels_syn = zeros(size(Theta_syn_all,1),1);  % preallocate

start_idx = 1;
for i = 1:length(unique_stress)
    M = size(Theta_syn_by_stress{i},1);  % actual number of synthetic samples for this stressor
    end_idx = start_idx + M - 1;
    
    if strcmp(unique_stress{i},'a_control')
        binary_labels_syn(start_idx:end_idx) = 0;
    else
        binary_labels_syn(start_idx:end_idx) = 1;
    end
    
    start_idx = end_idx + 1;
end

%% 6. Concatenate real + synthetic datasets

Theta_real = [];
valid_idx = false(length(AP_dataset),1);

for i = 1:length(AP_dataset)
    theta = AP_dataset{i}.theta;
    theta = theta(:)';  % enforce row shape
    
    if length(theta) == 5 && all(isfinite(theta))
        Theta_real = [Theta_real; theta];
        valid_idx(i) = true;
    end
end

Theta_all = [Theta_real; Theta_syn_all];
labels_all = [binary_labels_real; binary_labels_syn];

% 7. Save for ML
save('dataset_for_analysis.mat','Theta_all','labels_all');

disp(['Synthetic dataset created with ', num2str(size(Theta_all,1)), ' total APs']);
