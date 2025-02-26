import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# 1. .mat 파일 로드
matfile = r'/home/jyounglee/NL_real/Result/IWSDG_ALL.mat'
if not os.path.exists(matfile):
    raise FileNotFoundError(f'Error: {matfile} 파일이 현재 경로에 없습니다.')

data = loadmat(matfile)

# MATLAB 파일 내부의 불필요한 키를 제외하고, 실제 변수만 사용
if 'raw_spectra' not in data or 'denoised' not in data:
    raise KeyError(f'Error: {matfile} 파일 내에 raw_spectra와 denoised 변수가 모두 존재해야 합니다.')

raw = data['raw_spectra']   # shape: (1280, 1600)
den = data['denoised']      # shape: (1280, 1600)
numSpec, numPoints = raw.shape

# 2. x축 정의 (인덱스 1~1600에 대응하는 Raman shift)
#    0 -> -0.839, 1600 -> 3070.86, 길이 numPoints (1600)의 등간격 벡터
x = np.linspace(-0.839, 3070.86, numPoints)

# 3. 스펙트럼 선택 (MATLAB은 1-indexed이므로 Python에서는 인덱스를 1 빼줍니다)
spec_idx = 100  # MATLAB 상 100번째 스펙트럼 → Python에서는 index 99
if spec_idx > numSpec:
    raise ValueError(f'Error: 선택한 스펙트럼 인덱스가 전체 스펙트럼 수({numSpec})를 초과합니다.')

raw_spec = raw[spec_idx - 1, :]
den_spec = den[spec_idx - 1, :]

# 4. baseline 구간 및 피크 위치 (MATLAB 인덱스)
baseline_start = 1150
baseline_end   = 1400
peak_index     = 1109

if baseline_end > numPoints:
    raise ValueError(f'Error: baseline_end ({baseline_end})가 스펙트럼 길이({numPoints})를 초과합니다.')
if peak_index > numPoints:
    raise ValueError(f'Error: peak_index ({peak_index})가 스펙트럼 길이({numPoints})를 초과합니다.')

# Python에서는 0-indexed로 변환
baseline_start_idx = baseline_start - 1
baseline_end_idx = baseline_end      # Python slicing은 end-exclusive (즉, [baseline_start_idx:baseline_end_idx]는 MATLAB의 baseline_start ~ baseline_end와 동일)
peak_index_idx = peak_index - 1

# 5. baseline 추정 (raw 스펙트럼 기반, moving average)
windowSize = 20
smooth_baseline = np.convolve(raw_spec, np.ones(windowSize)/windowSize, mode='same')
# 이 baseline은 denoised 스펙트럼에도 동일하게 사용됩니다.

# 6. baseline 구간에서의 노이즈 수준 계산
noise_raw = raw_spec[baseline_start_idx:baseline_end_idx] - smooth_baseline[baseline_start_idx:baseline_end_idx]
noise_std_raw = np.std(noise_raw)

noise_den = den_spec[baseline_start_idx:baseline_end_idx] - smooth_baseline[baseline_start_idx:baseline_end_idx]
noise_std_den = np.std(noise_den)

# 7. 피크 위치에서의 SNR 계산
baseline_at_peak = smooth_baseline[peak_index_idx]
snr_raw = (raw_spec[peak_index_idx] - baseline_at_peak) / noise_std_raw
snr_den = (den_spec[peak_index_idx] - baseline_at_peak) / noise_std_den

# 8. 결과 출력
print(f"\n=== Spectrum #{spec_idx} Metrics (Single Baseline) ===")
print(f"Baseline Noise Std (Raw)      : {noise_std_raw:.6f}")
print(f"Baseline Noise Std (Denoised) : {noise_std_den:.6f}\n")
print(f"SNR (Raw)      : {snr_raw:.6f}")
print(f"SNR (Denoised) : {snr_den:.6f}")

# 9. 스펙트럼 비교 플롯 (x축은 실제 Raman shift)
plt.figure("Single Baseline Comparison", figsize=(10, 6))
plt.plot(x, raw_spec, color='c', linewidth=1.5, label='Raw')
plt.plot(x, den_spec, color='r', linewidth=1.5, label='Denoised')
plt.plot(x, smooth_baseline, 'b--', linewidth=1.5, label='Single Baseline (from raw)')

# baseline 구간 표시 (x값: baseline_start와 baseline_end)
plt.axvline(x=x[baseline_start_idx], color='g', linestyle='--', label='Baseline Start')
# Python slicing의 특성상, baseline_end_idx는 end-exclusive이므로, 인덱스 baseline_end_idx - 1 사용
plt.axvline(x=x[baseline_end_idx - 1], color='g', linestyle='--', label='Baseline End')

# 피크 위치 표시 (x값: peak_index)
plt.plot(x[peak_index_idx], raw_spec[peak_index_idx], 'bo', markersize=5, label='Raw Peak')
plt.plot(x[peak_index_idx], den_spec[peak_index_idx], 'ro', markersize=5, label='Denoised Peak')

plt.xlabel('Raman shift (cm$^{-1}$)')
plt.ylabel('Intensity')
plt.title(f'Spectrum #{spec_idx}: Single-Baseline Noise & SNR Comparison')
plt.grid(True)
plt.legend(loc='best')
plt.show()
