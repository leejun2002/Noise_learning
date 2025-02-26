% check_single_spectrum_metrics.m
% 하나의 baseline(원본 raw 스펙트럼으로부터 moving average)만 추정하여,
% raw와 denoised 모두에 대해 동일한 baseline을 기준으로
% 노이즈 수준(표준편차)와 특정 피크의 SNR을 계산해 비교합니다.
% * 가로축은 0~1600 인덱스를 -0.839 ~ 3070.86 (cm^-1)로 선형 매핑.

clear; clc; close all;

%% 1. .mat 파일 로드
matfile = 'IWSDG_coif4_6.mat';
if ~exist(matfile, 'file')
    error('Error: %s 파일이 현재 경로에 없습니다.', matfile);
end

data = load(matfile);

if ~isfield(data, 'raw_spectra') || ~isfield(data, 'denoised')
    error('Error: %s 파일 내에 raw_spectra와 denoised 변수가 모두 존재해야 합니다.', matfile);
end

raw = data.raw_spectra;    % (1280 x 1600)
den = data.denoised;       % (1280 x 1600)
[numSpec, numPoints] = size(raw);

%% 2. x축 정의 (인덱스 1~1600에 대응하는 Raman shift)
%   요구사항: 0 -> -0.839, 1600 -> 3070.86, 
%   길이가 numPoints(=1600)인 등간격 벡터
x = linspace(-0.839, 3070.86, numPoints);

%% 3. 스펙트럼 선택 (인덱스로)
spec_idx = 100;  % 1 ~ 1280 중 선택
if spec_idx > numSpec
    error('Error: 선택한 스펙트럼 인덱스가 전체 스펙트럼 수(%d)를 초과합니다.', numSpec);
end

raw_spec = raw(spec_idx, :);
den_spec = den(spec_idx, :);

%% 4. baseline 구간 및 피크 위치 (인덱스 단위로)
%  - baseline_start ~ baseline_end : 노이즈만 남아있다고 보는 영역 (인덱스)
%  - peak_index : 관심 피크 위치 (인덱스)
baseline_start = 1150;
baseline_end   = 1400;
peak_index     = 1109;

if baseline_end > numPoints
    error('Error: baseline_end (%d)가 스펙트럼 길이(%d)를 초과합니다.', baseline_end, numPoints);
end
if peak_index > numPoints
    error('Error: peak_index (%d)가 스펙트럼 길이(%d)를 초과합니다.', peak_index, numPoints);
end

%% 5. baseline 추정 (raw 스펙트럼 기반, moving average)
windowSize = 20; 
smooth_baseline = conv(raw_spec, ones(1, windowSize)/windowSize, 'same');
% => denoised 스펙트럼에도 같은 baseline 적용

%% 6. baseline 구간에서의 노이즈 수준
noise_raw = raw_spec(baseline_start:baseline_end) - ...
            smooth_baseline(baseline_start:baseline_end);
noise_std_raw = std(noise_raw);

noise_den = den_spec(baseline_start:baseline_end) - ...
            smooth_baseline(baseline_start:baseline_end);
noise_std_den = std(noise_den);

%% 7. 피크 위치에서의 SNR
baseline_at_peak = smooth_baseline(peak_index);

snr_raw = (raw_spec(peak_index) - baseline_at_peak) / noise_std_raw;
snr_den = (den_spec(peak_index) - baseline_at_peak) / noise_std_den;

%% 8. 결과 출력
fprintf('\n=== Spectrum #%d Metrics (Single Baseline) ===\n', spec_idx);
fprintf('Baseline Noise Std (Raw)      : %f\n', noise_std_raw);
fprintf('Baseline Noise Std (Denoised) : %f\n', noise_std_den);
fprintf('\nSNR (Raw)      : %f\n', snr_raw);
fprintf('SNR (Denoised) : %f\n', snr_den);

%% 9. 스펙트럼 비교 플롯 (x축은 실제 라만 시프트)
figure('Name','Single Baseline Comparison','NumberTitle','off');
plot(x, raw_spec, 'c-', 'LineWidth',1.5); hold on;
plot(x, den_spec, 'r-', 'LineWidth',1.5);
plot(x, smooth_baseline, 'b--', 'LineWidth',1.5);  % Raw 기반 baseline

% baseline 구간 표시 (x(baseline_start), x(baseline_end))
xline(x(baseline_start), '--g', 'Baseline Start');
xline(x(baseline_end), '--g', 'Baseline End');

% 피크 위치 표시 (x(peak_index))
plot(x(peak_index), raw_spec(peak_index), 'bo', 'MarkerSize',5, 'MarkerFaceColor','b');
plot(x(peak_index), den_spec(peak_index), 'ro', 'MarkerSize',5, 'MarkerFaceColor','r');

legend('Raw','Denoised','Single Baseline (from raw)', ...
       'Baseline Start/End','Raw Peak','Denoised Peak',...
       'Location','best');

xlabel('Raman shift (cm^{-1})');
ylabel('Intensity');
title(sprintf('Spectrum #%d: Single-Baseline Noise & SNR Comparison', spec_idx));
grid on;
hold off;
