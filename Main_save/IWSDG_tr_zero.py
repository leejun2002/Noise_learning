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

# Wavelet for baseline correction
import pywt

# 모델
from Model import UNet_CA, UNet, UNet_CA6, UNet_CA7
from Spectra_generator import spectra_generator
from Make_dataset import Read_data, Make_dataset
from config import DefaultConfig

EPS = 1e-6

###############################################################################
# helper functions
###############################################################################

def wavelet_baseline(y, wavelet=None, level=None):
    """
    Wavelet decomposition을 사용하여 baseline을 추정.
    detail 계수들을 모두 0으로 만들어 low-frequency 성분(= baseline)만 복원합니다.
    """
    coeffs = pywt.wavedec(y, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    baseline = pywt.waverec(coeffs, wavelet)
    return baseline[:len(y)]

def double_interpolation_cut(fpath, row_points=1600, zero_cut=0.0, remove_start=200):
    """
    1) 원본 스펙트럼(x,y)에 대해 x >= zero_cut 구간만 사용.
    2) 첫 번째 보간 → 1600 포인트.
    3) 앞부분 remove_start (기본=200) 포인트 제거 → 길이 1400.
    4) 두 번째 보간 → 다시 1600 포인트로 확장 후 반환.
    """
    try:
        arr = np.loadtxt(fpath)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        x, y = arr[:, 0], arr[:, 1]

        # 1) zero_cut 이하 데이터 제거
        mask = (x >= zero_cut)
        x_cut, y_cut = x[mask], y[mask]
        if len(x_cut) < 2:
            return None

        x_min, x_max = x_cut.min(), x_cut.max()
        if x_min == x_max:
            return None

        # 2) 첫 번째 보간 → 1600 포인트
        x_1600 = np.linspace(x_min, x_max, row_points)
        f1 = interp1d(x_cut, y_cut, kind='cubic', fill_value='extrapolate')
        y_1600 = f1(x_1600)

        # 3) 앞부분 remove_start 포인트 제거 (길이 1600 - remove_start = 1400)
        if remove_start >= row_points:
            return None
        y_1400 = y_1600[remove_start:]  # shape=(1400,)

        # 4) 두 번째 보간 → 다시 1600 포인트
        x_1400 = np.arange(len(y_1400))  # 0~1399
        x2_1600 = np.linspace(x_1400.min(), x_1400.max(), row_points)  # 0~1399 → 1600
        if len(x_1400) < 2:
            return None

        f2 = interp1d(x_1400, y_1400, kind='cubic', fill_value='extrapolate')
        y_final = f2(x2_1600)  # shape=(1600,)

        return y_final
    except Exception as e:
        print(f"[double_interpolation_cut] Error in {fpath}: {e}", flush=True)
        return None

def volume_interpolation(all_spectra, target_files=1280):
    """
    세로 방향(샘플 수) 보간.
    N개의 스펙트럼(각 길이 L)을 target_files=1280 개로 재보간 (파일 개수 방향)
    """
    N, L = all_spectra.shape
    old_idx = np.linspace(0, N - 1, N)
    new_idx = np.linspace(0, N - 1, target_files)
    out = np.zeros((target_files, L), dtype=all_spectra.dtype)
    for col in range(L):
        col_data = all_spectra[:, col]
        f_intp = interp1d(old_idx, col_data, kind='cubic', fill_value='extrapolate')
        out[:, col] = f_intp(new_idx)
    return out

def merge_txt_to_single_key_mat_1280(base_dir, out_mat, row_points=1600, target_files=1280):
    file_list = sorted(glob.glob(os.path.join(base_dir, '**', '*.txt'), recursive=True))
    if not file_list:
        print(f"[merge_txt] No .txt in {base_dir}")
        return None

    from multiprocessing import Pool
    from functools import partial
    worker = partial(double_interpolation_cut, row_points=row_points, zero_cut=0.0, remove_start=200)
    with Pool() as pool:
        results = pool.map(worker, file_list)
    results = [r for r in results if r is not None]
    if not results:
        print("[merge_txt] No valid spectra after interpolation/cut.")
        return None

    all_spectra = np.array(results, dtype=np.float64)
    print(f"[merge_txt] all_spectra shape = {all_spectra.shape}  # (파일 수, 1600)", flush=True)
    plt.figure("merge_txt: raw_spectra")
    plt.imshow(all_spectra, aspect='auto', cmap='jet')
    plt.title("After double_interpolation_cut (file direction not yet interpolated)")
    plt.colorbar()
    plt.show()

    data_matrix = volume_interpolation(all_spectra, target_files=target_files)
    print(f"[merge_txt] data_matrix shape = {data_matrix.shape}", flush=True)
    plt.figure("merge_txt: data_matrix")
    plt.imshow(data_matrix, aspect='auto', cmap='jet')
    plt.title("After volume_interpolation in file direction")
    plt.colorbar()
    plt.show()

    sio.savemat(out_mat, {"data_matrix": data_matrix})
    print(f"[merge_txt] Saved {out_mat}")
    return out_mat

###############################################################################
# train
###############################################################################
def train(config):
    base_folder = config.raw_noise_base
    subfolders = ("1", "2", "3")
    row_points = 1600
    target_files = 1280

    # 1. txt 파일 읽고 double_interpolation_cut, volume_interpolation 적용
    txt_files = []
    for sf in subfolders:
        p = os.path.join(base_folder, sf)
        fs = glob.glob(os.path.join(p, "*.txt"))
        txt_files.extend(fs)
    txt_files = sorted(txt_files)
    if not txt_files:
        print("[오류] no txt files.")
        return

    from multiprocessing import Pool
    from functools import partial
    worker = partial(double_interpolation_cut, row_points=row_points, zero_cut=0.0, remove_start=200)
    with Pool() as pool:
        results = pool.map(worker, txt_files)
    results = [r for r in results if r is not None]
    if not results:
        print("[오류] double_interpolation_cut failed or no valid data")
        return

    all_spectra = np.array(results, dtype=np.float64)  # shape: (파일 수, 1600)
    print(f"[train] after double_interpolation_cut shape = {all_spectra.shape}")
    plt.figure("train: after double_interpolation_cut")
    plt.plot(all_spectra[0])
    plt.title("Sample Spectrum after double_interpolation_cut (row direction)")
    plt.show()

    all_spectra = volume_interpolation(all_spectra, target_files=target_files)
    print(f"[train] after volume_interpolation shape = {all_spectra.shape}")
    plt.figure("train: after volume_interpolation")
    plt.imshow(all_spectra, aspect='auto', cmap='jet')
    plt.title("After volume_interpolation (file direction)")
    plt.colorbar()
    plt.show()

    # 2. wavelet baseline
    wavelet_model = config.wavelet_model
    wavelet_level = config.wavelet_level
    for i in range(all_spectra.shape[0]):
        orig = all_spectra[i, :]
        base = wavelet_baseline(orig, wavelet=wavelet_model, level=wavelet_level)
        all_spectra[i, :] = orig - base
    print("[train] wavelet baseline correction done.")

    # 3. SVD BG removal
    U, s, Vt = svd(all_spectra, full_matrices=False)
    s_mod = s.copy()
    remove_svs = 1
    for i in range(remove_svs):
        if i < len(s_mod):
            s_mod[i] *= config.fade_factor
    noise_processed = U @ np.diag(s_mod) @ Vt
    print("[train] after SVD BG removal:", noise_processed.shape)
    plt.figure("train: after SVD BG removal")
    plt.plot(noise_processed[0])
    plt.title("Sample Spectrum after SVD removal")
    plt.show()

    # 4. DCT global statistics
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    output_mat = os.path.join(config.train_data_root, f"noisE_{script_name}.mat")
    sio.savemat(output_mat, {"noise": noise_processed})

    noise_dct = np.zeros_like(noise_processed)
    for i in range(noise_processed.shape[0]):
        noise_dct[i, :] = dct(noise_processed[i, :], type=2, norm='ortho')
    global_mean = np.mean(noise_dct)
    global_std = np.std(noise_dct)
    print(f"[train] global_mean={global_mean:.4f}, global_std={global_std:.4f}")

    sio.savemat(output_mat, {
        "noise": noise_processed,
        "global_mean": global_mean,
        "global_std": global_std
    }, do_compression=True)

    plt.figure("train: DCT Histogram")
    plt.hist(noise_dct.flatten(), bins=50)
    plt.title("Histogram of DCT Coefficients")
    plt.show()

    # ------ model training pipeline ------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = eval(f"{config.model_name}(1,1)")
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    writer = SummaryWriter(config.logs)
    global_step = 0
    best_valid_loss = 1e9

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
            noise = noise.squeeze().numpy()  # (batch, spec)
            bsz, spec = noise.shape
            clean_spectra = gen_train.generator(spec, bsz).T
            noisy_spectra = clean_spectra + noise

            in_dct = np.zeros_like(noisy_spectra)
            tgt_dct = np.zeros_like(noise)
            for j in range(bsz):
                in_dct[j, :] = dct(noisy_spectra[j, :], type=2, norm='ortho')
                tgt_dct[j, :] = dct(noise[j, :], type=2, norm='ortho')
            in_norm = (in_dct - global_mean) / (global_std + EPS)
            tgt_norm = (tgt_dct - global_mean) / (global_std + EPS)
            in_norm = in_norm.reshape(bsz, 1, spec).astype(np.float32)
            tgt_norm = tgt_norm.reshape(bsz, 1, spec).astype(np.float32)

            inp_t = torch.from_numpy(in_norm).to(device)
            out_t = torch.from_numpy(tgt_norm).to(device)

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
        print(f"[Epoch {epoch}] Train Loss: {epoch_train_loss:.6f}")

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for noise_v in valid_loader:
                noise_v = noise_v.squeeze().numpy()
                bsz, spec = noise_v.shape
                clean_v = gen_valid.generator(spec, bsz).T
                noisy_v = clean_v + noise_v

                in_dct_v = np.zeros_like(noisy_v)
                tgt_dct_v = np.zeros_like(noise_v)
                for j in range(bsz):
                    in_dct_v[j, :] = dct(noisy_v[j, :], type=2, norm='ortho')
                    tgt_dct_v[j, :] = dct(noise_v[j, :], type=2, norm='ortho')

                in_norm_v = (in_dct_v - global_mean) / (global_std + EPS)
                tgt_norm_v = (tgt_dct_v - global_mean) / (global_std + EPS)
                in_norm_v = in_norm_v.reshape(bsz, 1, spec).astype(np.float32)
                tgt_norm_v = tgt_norm_v.reshape(bsz, 1, spec).astype(np.float32)

                inp_v = torch.from_numpy(in_norm_v).to(device)
                out_v = torch.from_numpy(tgt_norm_v).to(device)
                preds_v = model(inp_v)
                v_loss = nn.MSELoss()(preds_v, out_v)
                valid_loss += v_loss.item()

        valid_loss /= len(valid_loader)
        writer.add_scalar("valid_loss", valid_loss, global_step)
        scheduler.step(valid_loss)
        print(f"[Epoch {epoch}] Global Step: {global_step} | Valid Loss: {valid_loss:.6f}")

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
            print(f"[Save] Checkpoint saved at epoch {epoch+1}: {ckpt_path}")

    print(f"[train] Finished. best_valid_loss={best_valid_loss:.4f}")

###############################################################################
def batch_predict(config):
    import os, sys
    import glob
    import numpy as np
    import torch
    import scipy.io as sio
    from scipy.fftpack import dct, idct
    import matplotlib.pyplot as plt

    print("[batch_predict] Starting batch prediction...")

    # (1) double_interpolation_cut + volume_interpolation 실행
    row_points = 1600
    target_files = 1280
    base_dir = config.raw_predict_base
    out_mat = os.path.join(config.batch_predict_root, "batchPredict_data.mat")

    file_list = sorted(glob.glob(os.path.join(base_dir, '**', '*.txt'), recursive=True))
    if not file_list:
        print(f"[batch_predict] No .txt in {base_dir}")
        return

    from multiprocessing import Pool
    from functools import partial
    worker = partial(double_interpolation_cut, row_points=row_points, zero_cut=0.0, remove_start=200)
    with Pool() as pool:
        results = pool.map(worker, file_list)
    results = [r for r in results if r is not None]
    if not results:
        print("[batch_predict] double_interpolation_cut failed or no valid data")
        return

    raw_spectra = np.array(results, dtype=np.float64)  # shape: (파일 수, 1600)
    print("[batch_predict] after double_interpolation_cut:", raw_spectra.shape)

    data_matrix = volume_interpolation(raw_spectra, target_files=target_files)
    print("[batch_predict] after volume_interpolation:", data_matrix.shape)

    # (2) wavelet, SVD 등은 사용하지 않고 원본 그대로 -> dct -> model -> idct
    # train에서 구한 global_mean, global_std 불러오기
    train_script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    train_mat_path = os.path.join(config.train_data_root, f"noisE_{train_script_name}.mat")
    if not os.path.exists(train_mat_path):
        print(f"[batch_predict] Train mat not found: {train_mat_path}")
        return
    train_dict = sio.loadmat(train_mat_path, struct_as_record=False, squeeze_me=True)
    if 'global_mean' not in train_dict or 'global_std' not in train_dict:
        print("[batch_predict] global_mean/std not in train mat")
        return
    global_mean = float(train_dict['global_mean'])
    global_std = float(train_dict['global_std'])
    print(f"[batch_predict] global_mean={global_mean:.4f}, global_std={global_std:.4f}")

    # (3) 모델 불러오기
    model_file = os.path.join(config.test_model_dir, f"{config.global_step}.pt")
    if not os.path.exists(model_file):
        print(f"[batch_predict] Model file not found: {model_file}")
        return
    state = torch.load(model_file, map_location='cpu')
    model = eval(f"{config.model_name}(1,1)")
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # (4) 예측: DCT -> (normalize) -> model -> IDCT
    predicted_noise = np.zeros_like(data_matrix)
    N, L = data_matrix.shape
    for i in range(N):
        row_spec = data_matrix[i, :]
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

    # (5) 노이즈 제거
    denoised = data_matrix - predicted_noise
    print(f"[batch_predict] predicted_noise shape = {predicted_noise.shape}, denoised shape = {denoised.shape}")

    plt.figure("Batch Predict: Sample Compare")
    plt.plot(data_matrix[0], label="Raw Spec")
    plt.plot(predicted_noise[0], label="Pred Noise")
    plt.plot(denoised[0], label="Denoised")
    plt.legend()
    plt.show()

    # (6) 결과 저장
    out_dict = {
        "raw_spectra": data_matrix,
        "predicted_noise": predicted_noise,
        "denoised": denoised
    }
    sio.savemat(out_mat, out_dict, do_compression=True)
    print(f"[batch_predict] Saved {out_mat}")

###############################################################################
def check_dir(config):
    for d in [config.checkpoint, config.logs, config.test_dir]:
        if not os.path.exists(d):
            os.mkdir(d)

def save_model_dir(config):
    d = os.path.join(config.checkpoint, config.Instrument, config.model_name, f"batch_{config.batch_size}")
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def main(config):
    check_dir(config)
    if config.is_training:
        train(config)
    if config.is_batch_predicting:
        batch_predict(config)

if __name__ == "__main__":
    from config import DefaultConfig
    opt = DefaultConfig()
    opt.wavelet_model = "sym4"
    opt.wavelet_level = 4
    # ex) opt.is_training = True  # or False
    main(opt)
    plt.show()
