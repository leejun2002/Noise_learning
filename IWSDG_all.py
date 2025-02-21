import os
import glob
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial

from scipy.fftpack import dct, idct
from scipy.io import savemat
from scipy.interpolate import interp1d
from scipy.linalg import svd

import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'

from multiprocessing import Pool
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve

# Mixed Precision
from torch.cuda.amp import autocast, GradScaler

# 추가: PyWavelets for wavelet-based baseline correction
import pywt

# 모델 파일들 (예: U-Net 계열)
from Model import UNet_CA, UNet, UNet_CA6, UNet_CA7
from Spectra_generator import spectra_generator
from Make_dataset import Read_data, Make_dataset
from config import DefaultConfig

EPS = 1e-6

###############################################################################
# Global helper functions
###############################################################################

def wavelet_baseline(y, wavelet=None, level=None):
    """
    Wavelet decomposition을 사용한 baseline 추정.
    입력 신호 y에 대해 주어진 wavelet과 level로 분해한 후,
    detail 계수를 모두 0으로 만들어 low-frequency 성분만 복원합니다.
    """
    coeffs = pywt.wavedec(y, wavelet, level=level)
    # detail 계수들을 모두 0으로 설정
    coeffs[1:] = [np.zeros_like(detail) for detail in coeffs[1:]]
    baseline = pywt.waverec(coeffs, wavelet)
    return baseline[:len(y)]

def length_interpolation(fpath, row_points=1600, zero_cut=0.0):
    try:
        arr = np.loadtxt(fpath)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        x, y = arr[:, 0], arr[:, 1]
        mask = (x >= zero_cut)
        x_cut, y_cut = x[mask], y[mask]
        if len(x_cut) < 2:
            return None
        x_min, x_max = x_cut.min(), x_cut.max()
        if x_min == x_max:
            return None
        x_new = np.linspace(x_min, x_max, row_points)
        f_intp = interp1d(x_cut, y_cut, kind='cubic', fill_value='extrapolate')
        y_new = f_intp(x_new)
        return y_new
    except Exception as e:
        print(f"Interpolation error in {fpath}: {e}", flush=True)
        return None

def volume_interpolation(all_spectra, target_files=1280):
    N, L = all_spectra.shape
    old_idx = np.linspace(0, N - 1, N)
    new_idx = np.linspace(0, N - 1, target_files)
    out = np.zeros((target_files, L), dtype=all_spectra.dtype)
    for col in range(L):
        y_col = all_spectra[:, col]
        f_intp = interp1d(old_idx, y_col, kind='cubic', fill_value='extrapolate')
        out[:, col] = f_intp(new_idx)
    return out

def merge_txt_to_single_key_mat_1280(base_dir, out_mat, row_points=1600, target_files=1280):
    file_list = sorted(glob.glob(os.path.join(base_dir, '**', '*.txt'), recursive=True))
    if not file_list:
        print(f"[오류] '{base_dir}' 내에 .txt 파일이 없습니다.", flush=True)
        return None

    with Pool() as pool:
        worker_func = partial(length_interpolation, row_points=row_points, zero_cut=0.0)
        results = pool.map(worker_func, file_list)
    results = [r for r in results if r is not None]
    if not results:
        print("[오류] 유효한 스펙트럼이 없습니다.", flush=True)
        return None

    all_spectra = np.array(results, dtype=np.float64)
    print(f"[merge_txt] Initial shape from files: {all_spectra.shape}", flush=True)
    plt.figure("merge_txt: Raw Spectra")
    plt.imshow(all_spectra, aspect='auto', cmap='jet')
    plt.title("Raw Spectra (Before Volume Interpolation)")
    plt.colorbar()
    plt.show()

    data_matrix = volume_interpolation(all_spectra, target_files=target_files)
    print(f"[merge_txt] Final data_matrix shape: {data_matrix.shape} (target_files x {row_points})", flush=True)
    plt.figure("merge_txt: Data Matrix")
    plt.imshow(data_matrix, aspect='auto', cmap='jet')
    plt.title("Data Matrix after Volume Interpolation")
    plt.colorbar()
    plt.show()

    sio.savemat(out_mat, {"data_matrix": data_matrix})
    print(f"[merge_txt] Saved data_matrix to '{out_mat}'", flush=True)
    return out_mat

###############################################################################
# 로그, 체크포인트 등 저장 경로 함수
###############################################################################
def check_dir(config):
    for d in [config.checkpoint, config.logs, config.test_dir]:
        if not os.path.exists(d):
            os.mkdir(d)

def test_result_dir(config):
    rdir = os.path.join(config.batch_save_root, config.Instrument, config.model_name, "step_" + str(config.global_step))
    if not os.path.exists(rdir):
        os.makedirs(rdir)
    return rdir

def save_model_dir(config):
    d = os.path.join(config.checkpoint, config.Instrument, config.model_name, "batch_" + str(config.batch_size))
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def save_log_dir(config):
    # 첫번째 코드와 유사한 로그 저장 경로 지정
    log_dir = os.path.join(config.logs, config.Instrument, config.model_name, "batch_" + str(config.batch_size))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

###############################################################################
# train pipeline (includes noise processing and model training)
###############################################################################
def train(config):
    # 1. 전처리: noise data 생성
    base_folder = config.raw_noise_base
    subfolders = ("1", "2", "3")
    row_points = 1600
    target_files = 1280
    zero_cut = 0.0

    txt_files = []
    for sf in subfolders:
        sf_path = os.path.join(base_folder, sf)
        fs = glob.glob(os.path.join(sf_path, "*.txt"))
        txt_files.extend(fs)
    txt_files = sorted(txt_files)
    if not txt_files:
        print("[오류] Train txt 파일 없음.", flush=True)
        return

    with Pool() as pool:
        worker_func = partial(length_interpolation, row_points=row_points, zero_cut=zero_cut)
        results = pool.map(worker_func, txt_files)
    results = [r for r in results if r is not None]
    if not results:
        print("[오류] Row interpolation 실패.", flush=True)
        return
    all_spectra = np.array(results, dtype=np.float64)
    print(f"[train pipeline] After row interpolation: {all_spectra.shape}", flush=True)
    plt.figure("Train: After Row Interpolation")
    plt.plot(all_spectra[0])
    plt.title("Sample Spectrum after Row Interpolation")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.show()

    all_spectra = volume_interpolation(all_spectra, target_files=target_files)
    print(f"[train pipeline] After volume interpolation: {all_spectra.shape}", flush=True)
    plt.figure("Train: After Volume Interpolation")
    plt.imshow(all_spectra, aspect='auto', cmap='jet')
    plt.title("Data Matrix after Volume Interpolation")
    plt.colorbar()
    plt.show()

    # 2. Wavelet 기반 baseline 제거 (config.wavelet_model, config.wavelet_level 사용)
    baselines = []
    originals = []
    for i in range(all_spectra.shape[0]):
        orig = all_spectra[i, :].copy()
        baseline = wavelet_baseline(orig, wavelet=config.wavelet_model, level=config.wavelet_level)
        baselines.append(baseline)
        originals.append(orig)
        all_spectra[i, :] = orig - baseline

    print("Wavelet baseline correction applied.")
    plt.figure("Train: Wavelet Baseline Correction")
    i = 0  # 첫 번째 스펙트럼
    plt.plot(originals[i], label="Original (before baseline removal)")
    plt.plot(baselines[i], label="Wavelet Baseline")
    plt.plot(all_spectra[i, :], label="After baseline removal")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.title("Sample Spectrum: Original, Baseline, After Wavelet Correction")
    plt.legend()
    plt.show()

    # 3. SVD Background Removal
    U, s, Vt = svd(all_spectra, full_matrices=False)
    s_mod = s.copy()
    remove_svs = 1
    for i in range(remove_svs):
        if i < len(s_mod):
            s_mod[i] *= config.fade_factor
    noise_processed = U @ np.diag(s_mod) @ Vt
    print(f"[train pipeline] After SVD BG removal: {noise_processed.shape}", flush=True)
    plt.figure("Train: After SVD BG Removal")
    plt.plot(noise_processed[0])
    plt.title("Sample Spectrum after SVD Background Removal")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.show()

    # 4. Save noise data and compute global DCT statistics
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    output_mat = os.path.join(config.train_data_root, f"noisE_{script_name}.mat")
    sio.savemat(output_mat, {"noise": noise_processed})
    print(f"[train pipeline] Saved noisE_{script_name}.mat to {output_mat}", flush=True)
    noise_dct = np.zeros_like(noise_processed)
    for i in range(noise_processed.shape[0]):
        noise_dct[i, :] = dct(noise_processed[i, :], type=2, norm='ortho')
    global_mean = np.mean(noise_dct)
    global_std = np.std(noise_dct)
    print(f"[train pipeline] global_mean={global_mean:.4f}, global_std={global_std:.4f}", flush=True)
    sio.savemat(output_mat, {"noise": noise_processed,
                              "global_mean": global_mean,
                              "global_std": global_std}, do_compression=True)
    plt.figure("Train: DCT Histogram")
    plt.hist(noise_dct.flatten(), bins=50)
    plt.title("Histogram of DCT Coefficients")
    plt.xlabel("Coefficient value")
    plt.ylabel("Frequency")
    plt.show()

    # ----------------- Begin model training -----------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = eval(f"{config.model_name}(1,1)")
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    # 수정된 부분: SummaryWriter를 생성할 때, 로그 저장 경로 지정
    writer = SummaryWriter(save_log_dir(config))
    global_step = 0
    best_valid_loss = 1e9
    best_models = []

    reader = Read_data(config.train_data_root, config.valid_ratio)
    train_files, valid_files = reader.read_file()
    TrainSet, ValidSet = Make_dataset(train_files), Make_dataset(valid_files)
    train_loader = DataLoader(TrainSet, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(ValidSet, batch_size=256, pin_memory=True)
    gen_train = spectra_generator()
    gen_valid = spectra_generator()

    for epoch in range(config.max_epoch):
        model.train()
        epoch_train_loss = 0.0
        for idx, noise in enumerate(train_loader):
            noise = noise.squeeze().numpy()  # shape: (batch, spec)
            bsz, spec = noise.shape
            clean_spectra = gen_train.generator(spec, bsz).T  # (spec, bsz)
            noisy_spectra = clean_spectra + noise  # (spec, bsz)
            input_dct = np.zeros_like(noisy_spectra)
            target_dct = np.zeros_like(noise)
            for j in range(bsz):
                input_dct[j, :] = dct(noisy_spectra[j, :], type=2, norm='ortho')
                target_dct[j, :] = dct(noise[j, :], type=2, norm='ortho')
            input_norm = (input_dct - global_mean) / (global_std + EPS)
            target_norm = (target_dct - global_mean) / (global_std + EPS)
            input_norm = input_norm.reshape(bsz, 1, spec).astype(np.float32)
            target_norm = target_norm.reshape(bsz, 1, spec).astype(np.float32)

            inp_t = torch.from_numpy(input_norm).to(device)
            out_t = torch.from_numpy(target_norm).to(device)
            preds = model(inp_t)
            loss = nn.MSELoss()(preds, out_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            global_step += 1
            writer.add_scalar("train_loss", loss.item(), global_step)

        epoch_train_loss /= len(train_loader)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for noise_v in valid_loader:
                noise_v = noise_v.squeeze().numpy()
                bsz, spec = noise_v.shape
                clean_v = gen_valid.generator(spec, bsz).T
                noisy_v = clean_v + noise_v

                inp_dct = np.zeros_like(noisy_v)
                tgt_dct = np.zeros_like(noise_v)
                for j in range(bsz):
                    inp_dct[j, :] = dct(noisy_v[j, :], type=2, norm='ortho')
                    tgt_dct[j, :] = dct(noise_v[j, :], type=2, norm='ortho')
                inp_norm_v = (inp_dct - global_mean) / (global_std + EPS)
                tgt_norm_v = (tgt_dct - global_mean) / (global_std + EPS)
                inp_norm_v = inp_norm_v.reshape(bsz, 1, spec).astype(np.float32)
                tgt_norm_v = tgt_norm_v.reshape(bsz, 1, spec).astype(np.float32)
                inp_v = torch.from_numpy(inp_norm_v).to(device)
                out_v = torch.from_numpy(tgt_norm_v).to(device)
                with torch.no_grad():
                    preds_v = model(inp_v)
                    v_loss = nn.MSELoss()(preds_v, out_v)
                valid_loss += v_loss.item()
        valid_loss /= len(valid_loader)
        writer.add_scalar("valid_loss", valid_loss, global_step)
        scheduler.step(valid_loss)

        # 한 줄로 출력: epoch, global step, train loss, valid loss
        print(f"[Epoch {epoch}] Global Step: {global_step} | Train Loss: {epoch_train_loss:.6f} | Valid Loss: {valid_loss:.6f}", flush=True)

        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(save_model_dir(config), f"{global_step}.pt")
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_step': global_step,
                'epoch': epoch,
                'loss': loss.item()
            }
            torch.save(state, ckpt_path)
            print(f"[Save] Checkpoint saved at epoch {epoch+1}: {ckpt_path}", flush=True)

    # 학습이 끝난 후 SummaryWriter를 닫아 로그 파일을 완전히 저장합니다.
    writer.close()
    print(f"[train] Finished. Best valid loss = {best_valid_loss:.4f}", flush=True)

###############################################################################
# predict
###############################################################################

def predict(config, row_points=1600, zero_cut=0.0):
    """
    batch_predict 함수를 참고하여, SVD를 생략하고 Wavelet만 적용한 뒤
    (global_mean, global_std)로 정규화 → 모델 노이즈 예측 → raw - noise 로
    denoised를 구하는 파이프라인을 '단일 txt 파일' 단위로 처리하는 함수.

    1) config.predict_root 폴더 내의 모든 .txt 파일 찾기
    2) 각 파일에 대하여:
       - txt 로드 + length_interpolation
       - Wavelet Baseline 제거
       - DCT & (global_mean, global_std) 정규화
       - 모델 추론(노이즈)
       - 최종 denoised = raw - predicted_noise
       - (옵션) 시각화(2x2 subplot 등 batch_predict와 비슷하게)
       - .mat 파일로 결과 저장
    """

    print("[predict] Starting single-file prediction pipeline (Wavelet only, no SVD)...", flush=True)

    # ------------------------------------------------
    # 1) predict_root 폴더 내 모든 .txt 파일 찾기
    # ------------------------------------------------
    txt_files = sorted(glob.glob(os.path.join(config.predict_root, "*.txt")))
    if not txt_files:
        print(f"[predict] '{config.predict_root}' 폴더에 .txt 파일이 없습니다.")
        return

    # ------------------------------------------------
    # 2) 학습 시점 noise mat (noisE_{script_name}.mat)에서 global_mean, global_std 로드
    # ------------------------------------------------
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    train_noise_mat = os.path.join(config.train_data_root, f"noisE_{script_name}.mat")
    if not os.path.exists(train_noise_mat):
        print(f"[predict] '{train_noise_mat}' 파일이 존재하지 않습니다.")
        return

    train_dict = sio.loadmat(train_noise_mat, struct_as_record=False, squeeze_me=True)
    if 'global_mean' not in train_dict or 'global_std' not in train_dict:
        print("[predict] 'global_mean' 혹은 'global_std'를 찾을 수 없습니다.")
        return

    global_mean = float(train_dict['global_mean'])
    global_std  = float(train_dict['global_std'])
    EPS = 1e-6

    print(f"[predict] Loaded global_mean={global_mean:.6f}, global_std={global_std:.6f}")

    # ------------------------------------------------
    # 3) 모델 로드 (batch_predict와 동일하게)
    # ------------------------------------------------
    model_file = os.path.join(config.test_model_dir, f"{config.global_step}.pt")
    if not os.path.exists(model_file):
        print(f"[predict] 모델 파일을 찾을 수 없습니다: {model_file}")
        return

    state = torch.load(model_file, map_location='cpu')
    model = eval(f"{config.model_name}(1,1)")
    # DataParallel 저장 고려
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    print(f"[predict] Loaded model from {model_file}")

    # ------------------------------------------------
    # 4) txt 파일마다 처리
    # ------------------------------------------------
    for txt_file in txt_files:
        print(f"\n=== Processing {txt_file} ===")

        # (4-1) txt 로드 + interpolation
        spec = length_interpolation(txt_file, row_points=row_points, zero_cut=zero_cut)
        if spec is None:
            print(f"[predict] 보간 실패 or 데이터가 유효하지 않습니다: {txt_file}")
            continue

        # (4-2) Wavelet Baseline Correction
        # batch_predict처럼 orig - baseline
        baseline = wavelet_baseline(spec, wavelet=config.wavelet_model, level=config.wavelet_level)
        wavelet_corrected = spec - baseline

        # (4-3) DCT & Normalization
        row_dct = dct(wavelet_corrected, type=2, norm='ortho')
        row_norm = (row_dct - global_mean) / (global_std + EPS)

        # (4-4) 모델 추론 -> predicted_noise (DCT domain)
        inp_t = torch.from_numpy(row_norm.reshape(1,1,-1).astype(np.float32))
        if torch.cuda.is_available():
            inp_t = inp_t.cuda()
        with torch.no_grad():
            pred_out = model(inp_t).cpu().numpy().reshape(-1)

        # 역정규화 + IDCT
        pred_trans = pred_out * (global_std + EPS) + global_mean
        predicted_noise = idct(pred_trans, type=2, norm='ortho')

        # (4-5) 최종 denoised = raw - noise
        denoised = spec - predicted_noise

        # ------------------------------------------------
        # 5) 시각화: batch_predict와 비슷한 2x2 subplot
        # ------------------------------------------------
        fig, axs = plt.subplots(2, 2, figsize=(14, 9))
        axs = axs.flatten()

        # subplot(0): 원본 / wavelet
        axs[0].plot(spec, label="Original (Interpolated)", color='blue')
        axs[0].plot(baseline, label="Wavelet Baseline", color='orange')
        axs[0].plot(wavelet_corrected, label="After Wavelet", color='green')
        axs[0].set_title("Wavelet Baseline Correction")
        axs[0].set_xlabel("Index"); axs[0].set_ylabel("Intensity")
        axs[0].legend()

        # subplot(1): DCT & Normalized
        axs[1].plot(row_dct, label="DCT", color='royalblue')
        axs[1].plot(row_norm, label="Normalized DCT", color='red')
        axs[1].set_title("DCT & Normalized")
        axs[1].set_xlabel("Coeff Index"); axs[1].set_ylabel("Value")
        axs[1].legend()

        # subplot(2): Predicted Noise
        axs[2].plot(predicted_noise, label="Predicted Noise", color='purple')
        axs[2].set_title("Predicted Noise (Time Domain)")
        axs[2].set_xlabel("Index"); axs[2].set_ylabel("Intensity")
        axs[2].legend()

        # subplot(3): Raw vs Noise vs Denoised
        axs[3].plot(spec, label="Raw", color='cyan')
        axs[3].plot(predicted_noise, label="Pred Noise", color='goldenrod')
        axs[3].plot(denoised, label="Denoised", color='magenta')
        axs[3].set_title("Comparison (Raw / Noise / Denoised)")
        axs[3].set_xlabel("Index"); axs[3].set_ylabel("Intensity")
        axs[3].legend()

        plt.suptitle(f"Predict for: {os.path.basename(txt_file)}", fontsize=14)
        plt.tight_layout()
        plt.show()

        # ------------------------------------------------
        # 6) .mat 파일로 저장
        # ------------------------------------------------
        fname_no_ext = os.path.splitext(os.path.basename(txt_file))[0]
        out_mat_path = os.path.join(config.predict_save_root, f"predict_{fname_no_ext}.mat")

        out_dict = {
            "original_interpolated": spec,
            "wavelet_corrected": wavelet_corrected,
            "predicted_noise": predicted_noise,
            "denoised": denoised
        }
        sio.savemat(out_mat_path, out_dict, do_compression=True)
        print(f"[predict] Saved => {out_mat_path}")

    print(f"[predict] Finished all. (Total {len(txt_files)} files)")

###############################################################################
# batch_predict (using global helper functions)
###############################################################################
def batch_predict(config):
    import os, sys
    import glob
    import numpy as np
    import torch
    import scipy.io as sio
    from scipy.fftpack import dct, idct
    from scipy.linalg import svd
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt

    print("[batch_predict] Starting unified batch prediction pipeline...", flush=True)
    row_points = 1600
    target_files = 1280
    zero_cut = 0.0

    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    merged_mat = os.path.join(config.batch_predict_root, f"noisY_{script_name}.mat")
    out_mat = merge_txt_to_single_key_mat_1280(
        base_dir=config.raw_predict_base,
        out_mat=merged_mat,
        row_points=row_points,
        target_files=target_files
    )
    if out_mat is None or not os.path.exists(out_mat):
        print("[오류] merge_txt_to_single_key_mat_1280 failed.", flush=True)
        return

    tmp = sio.loadmat(out_mat, struct_as_record=False, squeeze_me=True)
    if 'data_matrix' not in tmp:
        print("[오류] data_matrix key not found.", flush=True)
        return
    raw_data_matrix = tmp['data_matrix'].astype(np.float64)
    print(f"[batch_predict] raw_data_matrix shape = {raw_data_matrix.shape}", flush=True)
    plt.figure("BatchPredict: Raw Spectrum")
    plt.plot(raw_data_matrix[0])
    plt.title("Sample Raw Spectrum")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.show()

    # Wavelet baseline correction
    processed_matrix = raw_data_matrix.copy()
    N, L = processed_matrix.shape
    baselines = []
    originals = []
    for i in range(N):
        orig = processed_matrix[i, :].copy()
        baseline = wavelet_baseline(orig, wavelet=config.wavelet_model, level=config.wavelet_level)
        baselines.append(baseline)
        originals.append(orig)
        processed_matrix[i, :] = orig - baseline

    plt.figure("BatchPredict: Wavelet Correction")
    i = 0
    plt.plot(originals[i], label="Original (before Wavelet)")
    plt.plot(baselines[i], label="Wavelet Baseline")
    plt.plot(processed_matrix[i, :], label="After Wavelet")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.title("Sample Spectrum: Wavelet Correction")
    plt.legend()
    plt.show()

    # SVD Background Removal
    U, s, Vt = svd(processed_matrix, full_matrices=False)
    s_mod = s.copy()
    remove_svs = 1
    fade_factor = config.fade_factor
    for i in range(remove_svs):
        if i < len(s_mod):
            s_mod[i] *= fade_factor
    data_bg_removed = U @ np.diag(s_mod) @ Vt
    print(f"[batch_predict] After SVD BG removal: {data_bg_removed.shape}", flush=True)
    plt.figure("BatchPredict: After SVD BG Removal")
    plt.plot(data_bg_removed[0])
    plt.title("Sample Spectrum after SVD BG Removal")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.show()

    # Load global normalization parameters from training noise data
    train_noise_mat = os.path.join(config.train_data_root, f"noisE_{script_name}.mat")
    if not os.path.exists(train_noise_mat):
        print("[오류] noise_data.mat not found.", flush=True)
        return
    train_dict = sio.loadmat(train_noise_mat, struct_as_record=False, squeeze_me=True)
    if 'global_mean' not in train_dict or 'global_std' not in train_dict:
        print("[오류] Global mean/std not found in noise_data.mat", flush=True)
        return
    global_mean = float(train_dict['global_mean'])
    global_std = float(train_dict['global_std'])
    print(f"[batch_predict] global_mean={global_mean:.4f}, global_std={global_std:.4f}", flush=True)

    # Load pretrained model
    model_file = os.path.join(config.test_model_dir, f"{config.global_step}.pt")
    if not os.path.exists(model_file):
        print(f"[오류] Model file not found: {model_file}", flush=True)
        return
    state = torch.load(model_file, map_location='cpu')
    model = eval(f"{config.model_name}(1,1)")
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # Prediction: For each spectrum, predict the noise
    predicted_noise = np.zeros_like(data_bg_removed)
    for i in range(N):
        row_spec = data_bg_removed[i, :]
        row_dct = dct(row_spec, type=2, norm='ortho')
        row_norm = (row_dct - global_mean) / (global_std + EPS)
        inp_t = torch.from_numpy(row_norm.reshape(1, 1, -1).astype(np.float32))
        if torch.cuda.is_available():
            inp_t = inp_t.cuda()
        with torch.no_grad():
            pred_out = model(inp_t).cpu().numpy().reshape(-1)
        pred_trans = pred_out * (global_std + EPS) + global_mean
        row_noise = idct(pred_trans, type=2, norm='ortho')
        predicted_noise[i, :] = row_noise

    denoised = raw_data_matrix - predicted_noise
    print(f"[batch_predict] predicted_noise shape: {predicted_noise.shape}, denoised shape: {denoised.shape}", flush=True)

    # --- 추가: 예측 이후의 모든 단계를 시각화 ---
    # sample index 0을 대상으로 각 단계별 결과를 시각화합니다.
    sample_idx = 0
    # SVD BG 제거된 스펙트럼
    row_spec = data_bg_removed[sample_idx, :]
    # DCT 계산
    row_dct = dct(row_spec, type=2, norm='ortho')
    # 정규화
    row_norm = (row_dct - global_mean) / (global_std + EPS)
    # 모델 예측 (DCT domain)
    inp_t = torch.from_numpy(row_norm.reshape(1, 1, -1).astype(np.float32))
    if torch.cuda.is_available():
        inp_t = inp_t.cuda()
    with torch.no_grad():
        pred_out = model(inp_t).cpu().numpy().reshape(-1)
    # 역정규화 (DCT domain)
    pred_trans = pred_out * (global_std + EPS) + global_mean
    # IDCT를 통해 noise 복원
    row_noise = idct(pred_trans, type=2, norm='ortho')
    # 원본 raw spectrum과의 차이를 이용해 denoised spectrum 계산
    denoised_sample = raw_data_matrix[sample_idx, :] - row_noise

    # 여러 단계를 subplot으로 시각화
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()

    # axs[0].plot(raw_data_matrix[sample_idx, :], color='blue')
    # axs[0].set_title("Raw Spectrum")
    # axs[0].set_xlabel("Index")
    # axs[0].set_ylabel("Intensity")

    # axs[1].plot(data_bg_removed[sample_idx, :], color='orange')
    # axs[1].set_title("SVD BG Removed Spectrum")
    # axs[1].set_xlabel("Index")
    # axs[1].set_ylabel("Intensity")

    axs[0].plot(row_dct, color='royalblue')
    axs[0].set_title("DCT Coefficients")
    axs[0].set_xlabel("Coefficient Index")
    axs[0].set_ylabel("Value")

    axs[1].plot(row_norm, color='red')
    axs[1].set_title("Normalized DCT")
    axs[1].set_xlabel("Coefficient Index")
    axs[1].set_ylabel("Normalized Value")

    axs[2].plot(pred_trans, label="Inverse Norm DCT", color='orange')
    axs[2].plot(pred_out, label="Predicted DCT", color='purple')
    axs[2].set_title("Predicted DCT Coefficients")
    axs[2].set_xlabel("Coefficient Index")
    axs[2].set_ylabel("Value")
    axs[2].legend()

    axs[3].plot(row_noise, label="Predicted Noise", color='purple')
    axs[3].plot(denoised_sample, label="Denoised Spectrum", color='lime')
    axs[3].set_title("Predicted Noise & Denoised Spectrum")
    axs[3].set_xlabel("Index")
    axs[3].set_ylabel("Intensity")
    axs[3].legend()


    plt.tight_layout()
    plt.show()
    # --- 여기까지 추가 시각화 ---

    # 최종 결과 plot (비교)
    plt.figure("BatchPredict: Comparison")
    plt.plot(raw_data_matrix[0], label="Raw", color='cyan')
    plt.plot(predicted_noise[0], label="Predicted Noise", color='goldenrod')
    plt.plot(denoised[0], label="Denoised", color='magenta')
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.title("Sample Comparison (Raw, Predicted Noise, Denoised)")
    plt.legend()
    plt.show()

    out_dir = config.batch_save_root
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    out_name = os.path.join(out_dir, f"{script_name}.mat")
    out_dict = {
        "raw_spectra": raw_data_matrix,
        "predicted_noise": predicted_noise,
        "denoised": denoised
    }
    sio.savemat(out_name, out_dict, do_compression=True)
    print(f"[batch_predict] Saved results to {out_name}", flush=True)

###############################################################################
# main
###############################################################################
def main(config):
    check_dir(config)
    if config.is_training:
        train(config)
    if config.is_predicting:
        predict(config)
    if config.is_batch_predicting:
        batch_predict(config)

if __name__ == '__main__':
    opt = DefaultConfig()
    opt.visualize_steps = True
    main(opt)
    plt.show()
