import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def hex2rgb(hexStr):
    """
    '#RRGGBB' 형식의 문자열을 받아 0~1 범위의 RGB 튜플로 변환합니다.
    """
    if hexStr.startswith('#'):
        hexStr = hexStr[1:]
    if len(hexStr) != 6:
        raise ValueError("Invalid hex color string")
    r = int(hexStr[0:2], 16)
    g = int(hexStr[2:4], 16)
    b = int(hexStr[4:6], 16)
    return (r / 255, g / 255, b / 255)

# .mat 파일 경로를 지정합니다.
matFile = r'/home/jyounglee/NL_real/Result/IWSDG_ALL.mat'  # 필요한 경우 경로를 조정하세요.

# .mat 파일 로드
data = loadmat(matFile)

# 필수 필드가 존재하는지 확인합니다.
required_keys = ['raw_spectra', 'predicted_noise', 'denoised']
for key in required_keys:
    if key not in data:
        raise ValueError(f"The .mat file must contain {key}.")

# 시각화할 샘플 인덱스 선택
# MATLAB에서는 258번째 샘플을 선택하였으므로, Python의 0-indexed 기준에서는 257입니다.
sample_idx = 257

# 각 스펙트럼은 1x1600 벡터라고 가정합니다.
raw_spectrum = data['raw_spectra'][sample_idx, :]
predicted_noise = data['predicted_noise'][sample_idx, :]
denoised_spectrum = data['denoised'][sample_idx, :]

# -------------------------
# 서브플롯을 이용한 시각화
# -------------------------
plt.figure()

plt.subplot(3, 1, 1)
plt.plot(raw_spectrum, color='k', linewidth=1.5)
plt.title(f'Raw Spectrum (Sample {sample_idx + 1})')
plt.xlabel('Spectral Point')
plt.ylabel('Intensity')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(predicted_noise, color='r', linewidth=1.5)
plt.title(f'Predicted Noise (Sample {sample_idx + 1})')
plt.xlabel('Spectral Point')
plt.ylabel('Intensity')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(denoised_spectrum, color='b', linewidth=1.5)
plt.title(f'Denoised Spectrum (Sample {sample_idx + 1})')
plt.xlabel('Spectral Point')
plt.ylabel('Intensity')
plt.grid(True)

plt.tight_layout()
plt.show()

# --------------------------------
# 오버레이 플롯을 이용한 시각화
# --------------------------------
rawColor = hex2rgb('#117a65')         # 파란색 계열 (실제로는 어두운 청록색)
noiseColor = hex2rgb('#DE3163')        # 붉은색 계열
denoisedColor = hex2rgb('#873600')     # 초록색 계열 (hex 코드에 따라 다를 수 있음)

plt.figure(figsize=(15, 12))
plt.plot(raw_spectrum, color=rawColor, linewidth=1.5)
plt.plot(predicted_noise, color=noiseColor, linewidth=1.5)
plt.plot(denoised_spectrum, color=denoisedColor, linewidth=2)
plt.title(f'Overlay of Spectra (Sample {sample_idx + 1})')
plt.xlabel('Spectral Point')
plt.ylabel('Intensity')
plt.legend(['Raw Spectrum', 'Predicted Noise', 'Denoised Spectrum'])
plt.grid(True)
plt.show()
