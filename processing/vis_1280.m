function rgb = hex2rgb(hexStr)
    % hexStr은 '#RRGGBB' 형식의 문자열
    if hexStr(1) == '#'
        hexStr = hexStr(2:end);
    end
    if numel(hexStr) ~= 6
        error('Invalid hex color string');
    end
    r = sscanf(hexStr(1:2),'%2x');
    g = sscanf(hexStr(3:4),'%2x');
    b = sscanf(hexStr(5:6),'%2x');
    rgb = double([r, g, b]) / 255;
end

%% Sample Visualization for batch_predict_result.mat
% This script loads the .mat file and visualizes one sample spectrum:
% raw_spectra, predicted_noise, and denoised.
clear; clc; close all;

% Specify the path to your .mat file
matFile = 'IWSDG_all_m.mat'; % Adjust path if needed

% Load the data
data = load(matFile);

% Check that the required fields exist
if ~isfield(data, 'raw_spectra') || ~isfield(data, 'predicted_noise') || ~isfield(data, 'denoised')
    error('The .mat file must contain raw_spectra, predicted_noise, and denoised.');
end

% Choose the sample index you want to visualize (e.g., first sample)
sample_idx = 258;  % Change this index to view a different sample

% Extract the sample spectra (each is a 1x1600 vector)
raw_spectrum      = data.raw_spectra(sample_idx, :);
predicted_noise   = data.predicted_noise(sample_idx, :);
denoised_spectrum = data.denoised(sample_idx, :);

%% Visualization using Subplots
figure;
subplot(3,1,1);
plot(raw_spectrum, 'k', 'LineWidth', 1.5);
title(['Raw Spectrum (Sample ', num2str(sample_idx), ')']);
xlabel('Spectral Point');
ylabel('Intensity');
grid on;

subplot(3,1,2);
plot(predicted_noise, 'r', 'LineWidth', 1.5);
title(['Predicted Noise (Sample ', num2str(sample_idx), ')']);
xlabel('Spectral Point');
ylabel('Intensity');
grid on;

subplot(3,1,3);
plot(denoised_spectrum, 'b', 'LineWidth', 1.5);
title(['Denoised Spectrum (Sample ', num2str(sample_idx), ')']);
xlabel('Spectral Point');
ylabel('Intensity');
grid on;

%% Visualization with Overlay Plot
rawColor = hex2rgb('#117a65');         % 파란색 계열
noiseColor = hex2rgb('#DE3163');        % 붉은색 계열
denoisedColor = hex2rgb('#873600');     % 초록색 계열

figure('Position', [100, 100, 1500, 1200]); % [left, bottom, width, height]
plot(raw_spectrum, 'Color', rawColor, 'LineWidth', 1.5); hold on;
plot(predicted_noise, 'Color', noiseColor, 'LineWidth', 1.5);
plot(denoised_spectrum, 'Color', denoisedColor, 'LineWidth', 2);
title(['Overlay of Spectra (Sample ', num2str(sample_idx), ')']);
xlabel('Spectral Point');
ylabel('Intensity');
legend('Raw Spectrum', 'Predicted Noise', 'Denoised Spectrum');
grid on;
