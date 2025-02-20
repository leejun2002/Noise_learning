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
    print(f"[merge_txt] Final data_matrix shape: {data_matrix.shape} (target_files x {data_matrix.shape[1]})", flush=True)
    plt.figure("merge_txt: Data Matrix")
    plt.imshow(data_matrix, aspect='auto', cmap='jet')
    plt.title("Data Matrix after Volume Interpolation")
    plt.colorbar()
    plt.show()

    sio.savemat(out_mat, {"data_matrix": data_matrix})
    print(f"[merge_txt] Saved data_matrix to '{out_mat}'", flush=True)
    return out_mat

###############################################################################
# train pipeline (includes noise processing and model training)
###############################################################################
def train(config):
    base_folder = config.raw_noise_base
    subfolders = ("1", "2", "3")
    row_points = 1600
    target_files = 1280
    zero_cut = 0.0

    # 1. txt 파일 읽고 length_interpolation 적용
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
    worker = partial(length_interpolation, row_points=row_points, zero_cut=zero_cut)
    with Pool() as pool:
        results = pool.map(worker, txt_files)
    results = [r for r in results if r is not None]
    if not results:
        print("[오류] Row interpolation 실패.", flush=True)
        return

    all_spectra = np.array(results, dtype=np.float64)  # shape: (파일 수, 1600)
    print(f"[train] After row interpolation: {all_spectra.shape}")
    plt.figure("train: After Row Interpolation")
    plt.plot(all_spectra[0])
    plt.title("Sample Spectrum after Row Interpolation")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.show()

    # 2. volume interpolation (파일 수 방향 보간)
    all_spectra = volume_interpolation(all_spectra, target_files=target_files)
    print(f"[train] After volume interpolation: {all_spectra.shape}")
    plt.figure("train: After Volume Interpolation")
    plt.imshow(all_spectra, aspect='auto', cmap='jet')
    plt.title("Data Matrix after Volume Interpolation")
    plt.colorbar()
    plt.show()

    # 3. Wavelet 기반 baseline 제거
    wavelet_model = config.wavelet_model
    wavelet_level = config.wavelet_level
    for i in range(all_spectra.shape[0]):
        orig = all_spectra[i, :].copy()
        base = wavelet_baseline(orig, wavelet=wavelet_model, level=wavelet_level)
        all_spectra[i, :] = orig - base
    print("[train] Wavelet baseline correction applied.")
    plt.figure("train: Wavelet Baseline Correction")
    i = 0
    plt.plot(orig, label="Original (before baseline removal)")
    plt.plot(base, label="Wavelet Baseline")
    plt.plot(all_spectra[i, :], label="After baseline removal")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.title("Sample Spectrum: Original, Baseline, After Correction")
    plt.legend()
    plt.show()

    # 4. SVD Background Removal
    U, s, Vt = svd(all_spectra, full_matrices=False)
    s_mod = s.copy()
    remove_svs = 1
    for i in range(remove_svs):
        if i < len(s_mod):
            s_mod[i] *= config.fade_factor
    noise_processed = U @ np.diag(s_mod) @ Vt
    print(f"[train] After SVD BG removal: {noise_processed.shape}")
    plt.figure("train: After SVD BG Removal")
    plt.plot(noise_processed[0])
    plt.title("Sample Spectrum after SVD BG Removal")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.show()

    # 5. Instance normalization via DCT per sample (compute DCT, then normalize each sample individually)
    #    (global normalization 대신 각 샘플마다 평균과 std를 계산)
    noise_dct = np.zeros_like(noise_processed)
    inst_means = np.zeros((noise_processed.shape[0], 1))
    inst_stds = np.zeros((noise_processed.shape[0], 1))
    for i in range(noise_processed.shape[0]):
        noise_dct[i, :] = dct(noise_processed[i, :], type=2, norm='ortho')
        inst_means[i] = np.mean(noise_dct[i, :])
        inst_stds[i] = np.std(noise_dct[i, :])
    # 여기서는 global 통계 대신 instance statistics 사용 (혹은 둘 다 비교 가능)
    # 이 예제에서는 instance normalization만 사용
    # (DCT histogram 등을 위해 여전히 plot 가능)
    plt.figure("train: Instance DCT Histogram")
    plt.hist(noise_dct.flatten(), bins=50)
    plt.title("Histogram of DCT Coefficients (Instance Norm)")
    plt.xlabel("Coefficient value")
    plt.ylabel("Frequency")
    plt.show()

    # 저장 시에는 instance normalization을 사용하지 않으므로 noise_processed 그대로 저장
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    output_mat = os.path.join(config.train_data_root, f"noisE_{script_name}.mat")
    sio.savemat(output_mat, {"noise": noise_processed})
    print(f"[train] Saved noise data to {output_mat}")

    # ------ Begin model training ------
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
            noise = noise.squeeze().numpy()  # shape: (batch, spec)
            bsz, spec = noise.shape
            clean_spectra = gen_train.generator(spec, bsz).T  # (spec, bsz)
            noisy_spectra = clean_spectra + noise

            in_dct = np.zeros_like(noisy_spectra)
            tgt_dct = np.zeros_like(noise)
            # Instance normalization for each sample in the batch
            for j in range(bsz):
                in_dct[j, :] = dct(noisy_spectra[j, :], type=2, norm='ortho')
                tgt_dct[j, :] = dct(noise[j, :], type=2, norm='ortho')
            in_mean = np.mean(in_dct, axis=1, keepdims=True)
            in_std = np.std(in_dct, axis=1, keepdims=True)
            tgt_mean = np.mean(tgt_dct, axis=1, keepdims=True)
            tgt_std = np.std(tgt_dct, axis=1, keepdims=True)
            in_norm = (in_dct - in_mean) / (in_std + EPS)
            tgt_norm = (tgt_dct - tgt_mean) / (tgt_std + EPS)
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
                in_mean_v = np.mean(in_dct_v, axis=1, keepdims=True)
                in_std_v = np.std(in_dct_v, axis=1, keepdims=True)
                tgt_mean_v = np.mean(tgt_dct_v, axis=1, keepdims=True)
                tgt_std_v = np.std(tgt_dct_v, axis=1, keepdims=True)
                in_norm_v = (in_dct_v - in_mean_v) / (in_std_v + EPS)
                tgt_norm_v = (tgt_dct_v - tgt_mean_v) / (tgt_std_v + EPS)
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

    print(f"[train] Finished. Best valid loss = {best_valid_loss:.4f}")

###############################################################################
def batch_predict(config):
    import os, sys
    import glob
    import numpy as np
    import torch
    import scipy.io as sio
    from scipy.fftpack import dct, idct
    import matplotlib.pyplot as plt

    print("[batch_predict] Starting batch prediction pipeline...")

    row_points = 1600
    target_files = 1280
    zero_cut = 0.0

    # 1. 모든 .txt 파일들을 병합하여 data_matrix 생성
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    merged_mat = os.path.join(config.batch_predict_root, f"noisY_{script_name}.mat")
    out_mat = merge_txt_to_single_key_mat_1280(
        base_dir=config.raw_predict_base,
        out_mat=merged_mat,
        row_points=row_points,
        target_files=target_files
    )
    if out_mat is None or not os.path.exists(out_mat):
        print("[batch_predict] merge_txt_to_single_key_mat_1280 failed.", flush=True)
        return

    tmp = sio.loadmat(out_mat, struct_as_record=False, squeeze_me=True)
    if 'data_matrix' not in tmp:
        print("[batch_predict] data_matrix key not found.", flush=True)
        return
    raw_data_matrix = tmp['data_matrix'].astype(np.float64)
    print(f"[batch_predict] raw_data_matrix shape = {raw_data_matrix.shape}", flush=True)
    
    plt.figure("BatchPredict: Raw Spectrum")
    plt.plot(raw_data_matrix[0])
    plt.title("Sample Raw Spectrum")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.show()

    # 여기서는 wavelet baseline이나 SVD BG 제거 없이 원본 데이터를 그대로 사용
    processed_matrix = raw_data_matrix.copy()

    # 3. 모델 불러오기 및 예측
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

    # 2. 각 스펙트럼에 대해 DCT → Instance Normalization → 모델 예측 → IDCT
    predicted_noise = np.zeros_like(processed_matrix)
    N, L = processed_matrix.shape
    for i in range(N):
        row_spec = processed_matrix[i, :]
        row_dct = dct(row_spec, type=2, norm='ortho')
        # Instance normalization: 각 행마다 평균, std 계산
        row_mean = np.mean(row_dct)
        row_std = np.std(row_dct)
        row_norm = (row_dct - row_mean) / (row_std + EPS)
        inp_t = torch.from_numpy(row_norm.reshape(1, 1, -1).astype(np.float32))
        if torch.cuda.is_available():
            inp_t = inp_t.cuda()
        with torch.no_grad():
            pred_out = model(inp_t).cpu().numpy().reshape(-1)
        # Denormalize using same instance statistics
        pred_trans = pred_out * (row_std + EPS) + row_mean
        row_noise = idct(pred_trans, type=2, norm='ortho')
        predicted_noise[i, :] = row_noise

    # 3. Noise 제거: 원본 스펙트럼에서 예측된 noise를 빼서 denoised 결과 생성
    denoised = raw_data_matrix - predicted_noise
    print(f"[batch_predict] predicted_noise shape: {predicted_noise.shape}, denoised shape: {denoised.shape}", flush=True)
    
    plt.figure("BatchPredict: Comparison")
    plt.plot(raw_data_matrix[0], label="Raw")
    plt.plot(predicted_noise[0], label="Predicted Noise")
    plt.plot(denoised[0], label="Denoised")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.title("Sample Comparison")
    plt.legend()
    plt.show()

    # 4. 결과 저장
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
    main(opt)
    plt.show()
