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

# 모델 파일들 (예: U-Net 계열)
from Model import UNet_CA, UNet, UNet_CA6, UNet_CA7
from Spectra_generator import spectra_generator
from Make_dataset import Read_data, Make_dataset
from config import DefaultConfig

EPS = 1e-6

###############################################################################
# Global helper functions
###############################################################################

def als_baseline_sparse(y, lam=1e5, p=0.01, niter=10):
    """
    Sparse matrix-based ALS baseline correction.
    """
    L = len(y)
    main_diag = np.full(L, 2.0)
    off_diag = np.full(L - 1, -1.0)
    D2 = diags([main_diag, off_diag, off_diag], [0, -1, 1], shape=(L, L), format='csc')
    A_regular = lam * D2
    w = np.ones(L, dtype=np.float64)
    Z = np.zeros(L, dtype=np.float64)
    for _ in range(niter):
        W = diags(w, 0, shape=(L, L), format='csc')
        A_sp = A_regular + W
        b = w * y
        Z = spsolve(A_sp, b)
        w = np.where(y > Z, p, 1 - p)
    return Z

def length_interpolation(fpath, row_points=1600, zero_cut=0.0):
    """
    Reads a single txt file and interpolates its (x,y) spectrum along x
    to a fixed length (row_points) using cubic interpolation.
    """
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
    """
    Interpolates along the file (volume) axis so that the number of spectra becomes target_files.
    Uses cubic interpolation.
    Input shape: (N, row_points) and output shape: (target_files, row_points)
    """
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
    """
    Searches recursively under base_dir for all .txt files, then
    interpolates each (x, y) spectrum using cubic interpolation so that
    each spectrum has length = row_points. Then, uses volume interpolation
    to ensure the number of spectra equals target_files.
    Saves the resulting data_matrix (target_files x row_points) into a .mat file.
    """
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
    if True:
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
    all_spectra = np.array(results, dtype=np.float64)  # shape=(N, 1600)
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

    # 2. ALS Baseline Correction
    baselines = []
    originals = []
    for i in range(all_spectra.shape[0]):
        orig = all_spectra[i, :].copy()
        baseline = als_baseline_sparse(orig, lam=1e5, p=0.01, niter=10)
        baselines.append(baseline)
        originals.append(orig)
        all_spectra[i, :] = orig - baseline

    print("ALS baseline correction applied.")
    plt.figure("Train: ALS Correction")
    i = 0  # 첫 번째 스펙트럼 시각화
    plt.plot(originals[i], label="Original (before ALS)")
    plt.plot(baselines[i], label="ALS Baseline")
    plt.plot(all_spectra[i, :], label="After ALS")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.title("Sample Spectrum: Original, Baseline, After ALS")
    plt.legend()
    plt.show()

    # 3. 2D SVD Background Removal
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
        model = model.cuda()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    writer = SummaryWriter(config.logs)
    global_step = 0

    scaler = GradScaler()

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
            # Convert to (bsz, spec)
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

            with autocast():
                preds = model(inp_t)
                loss = nn.MSELoss()(preds, out_t)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()
            global_step += 1
            writer.add_scalar("train_loss", loss.item(), global_step)

        epoch_train_loss /= len(train_loader)
        print(f"[Epoch {epoch}] Train Loss: {epoch_train_loss:.6f}", flush=True)

        # Validation
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
                with autocast():
                    preds_v = model(inp_v)
                    v_loss = nn.MSELoss()(preds_v, out_v)
                valid_loss += v_loss.item()
        valid_loss /= len(valid_loader)
        writer.add_scalar("valid_loss", valid_loss, global_step)
        scheduler.step(valid_loss)
        print(f"[Epoch {epoch}] Global Step: {global_step} | Valid Loss: {valid_loss:.6f}", flush=True)

        # 50 epoch마다 시각화 (여기서는 매 epoch마다 예시)
        if config.visualize_steps:
            plt.figure(f"Epoch {epoch} Sample Prediction")
            plt.plot(clean_spectra[0, :], label="Clean Spectrum", color="blue")
            plt.plot(noisy_spectra[0, :], label="Noisy Spectrum", color="orange")
            plt.title(f"Sample Prediction at Epoch {epoch}")
            plt.xlabel("Index")
            plt.ylabel("Intensity")
            plt.legend()
            plt.show()

        # Checkpoint saving every 10 epochs
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

    print("[train] Finished training.")

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

    # (A) Merge txt files
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

    # (B) ALS and 2D SVD (CPU)
    processed_matrix = raw_data_matrix.copy()
    N, L = processed_matrix.shape
    baselines = []
    originals = []
    for i in range(N):
        orig = processed_matrix[i, :].copy()
        baseline = als_baseline_sparse(orig, lam=1e5, p=0.01, niter=10)
        baselines.append(baseline)
        originals.append(orig)
        processed_matrix[i, :] = orig - baseline

    plt.figure("BatchPredict: ALS Correction")
    i = 0
    plt.plot(originals[i], label="Original (before ALS)")
    plt.plot(baselines[i], label="ALS Baseline")
    plt.plot(processed_matrix[i, :], label="After ALS")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.title("Sample Spectrum: ALS Correction")
    plt.legend()
    plt.show()

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

    # (C) Load global DCT statistics from training file
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

    # (D) Load model
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

    # (E) DCT, predict noise, inverse DCT (row-wise)
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

    # (F) Denoised spectra
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

    # (G) Save results
    out_dir = config.batch_save_root
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = os.path.join(out_dir, f"{script_name}.mat")
    out_dict = {
        "raw_spectra": raw_data_matrix,
        "predicted_noise": predicted_noise,
        "denoised": denoised
    }
    sio.savemat(out_name, out_dict, do_compression=True)
    print(f"[batch_predict] Saved results to {out_name}", flush=True)

###############################################################################
# 경로 및 기타 함수들
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
    d = os.path.join(config.logs, config.Instrument, config.model_name, "batch_" + str(config.batch_size))
    if not os.path.exists(d):
        os.makedirs(d)
    return d

###############################################################################
# main
###############################################################################
def main(config):
    check_dir(config)
    if config.is_training:
        train(config)
    if config.is_batch_predicting:
        batch_predict(config)

if __name__ == '__main__':
    opt = DefaultConfig()
    opt.visualize_steps = True  # 시각화 활성화
    main(opt)
    plt.show()
