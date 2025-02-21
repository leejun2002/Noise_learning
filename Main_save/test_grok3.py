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
from scipy.io import savemat, loadmat
from scipy.interpolate import interp1d
from scipy.linalg import svd
import scipy.io as sio
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pywt
import logging

# PyTorch Mixed Precision
from torch.amp import autocast, GradScaler

# 사용자 제공 모듈
from CBAM1D import CBAM
from Model import UNet_CA, UNet, UNet_CA6, UNet_CA7, Conv_block_CA, Up_conv_CA
from Spectra_generator import spectra_generator
from Make_dataset import Read_data, Make_dataset
from config import DefaultConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO, filename="noise_learning.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EPS = 1e-6

###############################################################################
# Global Helper Functions
###############################################################################

def wavelet_baseline(y, wavelet='coif4', level=6):
    try:
        coeffs = pywt.wavedec(y, wavelet, level=level)
        coeffs[1:] = [np.zeros_like(detail) for detail in coeffs[1:]]
        return pywt.waverec(coeffs, wavelet)[:len(y)]
    except Exception as e:
        logger.error(f"Wavelet baseline error: {e}")
        return np.zeros_like(y)

def length_interpolation(fpath, row_points=1600, zero_cut=0.0):
    try:
        arr = np.loadtxt(fpath)
        if arr.ndim != 2 or arr.shape[1] < 2:
            logger.warning(f"Invalid data dimension in {fpath}")
            return None
        x, y = arr[:, 0], arr[:, 1]
        mask = (x >= zero_cut)
        x_cut, y_cut = x[mask], y[mask]
        if len(x_cut) < 2:
            logger.warning(f"Insufficient data points in {fpath}")
            return None
        x_new = np.linspace(x_cut.min(), x_cut.max(), row_points)
        f_intp = interp1d(x_cut, y_cut, kind='cubic', fill_value='extrapolate')
        return f_intp(x_new)
    except Exception as e:
        logger.error(f"Interpolation error in {fpath}: {e}")
        return None

def volume_interpolation(all_spectra, target_files=1280):
    N, L = all_spectra.shape
    old_idx = np.linspace(0, N - 1, N)
    new_idx = np.linspace(0, N - 1, target_files)
    out = np.zeros((target_files, L), dtype=all_spectra.dtype)
    for col in range(L):
        f_intp = interp1d(old_idx, all_spectra[:, col], kind='cubic', fill_value='extrapolate')
        out[:, col] = f_intp(new_idx)
    return out

def process_batch_dct(spectra, norm='ortho'):
    return dct(spectra, type=2, norm=norm, axis=1)

def process_batch_idct(coeffs, norm='ortho'):
    return idct(coeffs, type=2, norm=norm, axis=1)

def merge_txt_to_single_key_mat(base_dir, out_mat, row_points=1600, target_files=1280):
    file_list = sorted(glob.glob(os.path.join(base_dir, '**', '*.txt'), recursive=True))
    if not file_list:
        logger.error(f"No .txt files found in {base_dir}")
        return None
    
    print(f"Found {len(file_list)} files for merging")
    # 멀티프로세싱 대신 단일 스레드 처리 (필요 시 주석 해제)
    results = [length_interpolation(f, row_points=row_points, zero_cut=0.0) for f in file_list]
    # with Pool() as pool:
    #     worker_func = partial(length_interpolation, row_points=row_points, zero_cut=0.0)
    #     results = pool.map(worker_func, file_list)
    
    results = [r for r in results if r is not None]
    if not results:
        logger.error("No valid spectra after interpolation")
        return None

    all_spectra = np.zeros((len(results), row_points), dtype=np.float64)
    for i, r in enumerate(results):
        all_spectra[i] = r
    data_matrix = volume_interpolation(all_spectra, target_files=target_files)
    savemat(out_mat, {"data_matrix": data_matrix})
    logger.info(f"Saved data_matrix to {out_mat}, shape: {data_matrix.shape}")
    return out_mat

###############################################################################
# Directory Management Functions
###############################################################################

def check_dir(config):
    for d in [config.checkpoint, config.logs, config.test_dir]:
        os.makedirs(d, exist_ok=True)
    print("Directories checked and created if necessary")

def test_result_dir(config):
    rdir = os.path.join(config.batch_save_root, config.Instrument, config.model_name, f"step_{config.global_step}")
    os.makedirs(rdir, exist_ok=True)
    return rdir

def save_model_dir(config):
    d = os.path.join(config.checkpoint, config.Instrument, config.model_name, f"batch_{config.batch_size}")
    os.makedirs(d, exist_ok=True)
    return d

def save_log_dir(config):
    log_dir = os.path.join(config.logs, config.Instrument, config.model_name, f"batch_{config.batch_size}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

###############################################################################
# Training Pipeline
###############################################################################

def train(config):
    import scipy.io as sio
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    model = eval(f"{config.model_name}(1,1)").to(device)
    if device.type == "cuda" and torch.__version__ >= "2.0":
        model = torch.compile(model)
    print("Model initialized", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    writer = SummaryWriter(config.logs)

    # 데이터 준비
    base_folder = config.raw_noise_base
    subfolders = ("1", "2", "3")
    txt_files = sorted([f for sf in subfolders for f in glob.glob(os.path.join(base_folder, sf, "*.txt"))])
    if not txt_files:
        print("[오류] Train txt 파일 없음.", flush=True)
        return
    print(f"Found {len(txt_files)} training files", flush=True)

    with Pool() as pool:
        worker_func = partial(length_interpolation, row_points=1600, zero_cut=0.0)
        results = pool.map(worker_func, txt_files)
    results = [r for r in results if r is not None]
    all_spectra = np.array(results, dtype=np.float64)
    all_spectra = volume_interpolation(all_spectra, target_files=1280)
    if config.wavelet_model and config.wavelet_level:
        for i in range(all_spectra.shape[0]):
            baseline = wavelet_baseline(all_spectra[i, :], wavelet=config.wavelet_model, level=config.wavelet_level)
            all_spectra[i, :] -= baseline
    U, s, Vt = svd(all_spectra, full_matrices=False)
    s_mod = s.copy()
    s_mod[0] *= config.fade_factor
    noise_processed = U @ np.diag(s_mod) @ Vt
    noise_dct = process_batch_dct(noise_processed)  # CPU에서 한 번만 계산
    global_mean, global_std = np.mean(noise_dct), np.std(noise_dct)
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    output_mat = os.path.join(config.train_data_root, f"noise_{script_name}.mat")
    sio.savemat(output_mat, {"noise": noise_processed, "global_mean": global_mean, "global_std": global_std})
    print(f"Saved noise data to {output_mat}, mean={global_mean:.4f}, std={global_std:.4f}", flush=True)

    del all_spectra, noise_processed, noise_dct
    import gc
    gc.collect()

    # 데이터 로더
    reader = Read_data(config.train_data_root, config.valid_ratio)
    train_files, valid_files = reader.read_file()
    train_loader = DataLoader(Make_dataset(train_files), batch_size=32, shuffle=True, pin_memory=True, num_workers=4)
    valid_loader = DataLoader(Make_dataset(valid_files), batch_size=32, pin_memory=True, num_workers=4)
    print(f"Train loader: {len(train_loader)} batches, Valid loader: {len(valid_loader)} batches", flush=True)
    
    gen_train = spectra_generator()

    # GPU 기반 함수
    def batch_interpolation_gpu(noise, orig_points=1280, target_points=1600, device=device):
        noise_t = torch.from_numpy(noise).float().to(device)
        noise_interp = torch.nn.functional.interpolate(
            noise_t.unsqueeze(0), size=target_points, mode='linear', align_corners=False
        ).squeeze(0)
        return noise_interp

    def process_batch_dct_gpu(spectra, device=device):
        spectra_t = spectra if torch.is_tensor(spectra) else torch.from_numpy(spectra).float().to(device)
        fft = torch.fft.rfft(spectra_t, norm='ortho')
        N = spectra_t.shape[-1]
        dct = fft.real * torch.cos(torch.arange(N//2 + 1, device=device) * torch.pi / (2 * N))
        return dct

    # 학습 루프
    global_step, best_valid_loss = 0, float('inf')
    for epoch in range(config.max_epoch):
        model.train()
        epoch_train_loss = 0.0
        epoch_start = time.time()
        print(f"Starting epoch {epoch}/{config.max_epoch}", flush=True)
        for idx, noise in enumerate(train_loader):
            if idx % 10 == 0:
                print(f"Processing train batch {idx+1}/{len(train_loader)}", flush=True)
            try:
                t0 = time.time()
                noise = noise.squeeze().to(device)  # 바로 GPU로 로드
                bsz = noise.shape[0]
                
                t1 = time.time()
                clean_spectra = torch.from_numpy(gen_train.generator(1600, bsz).T).to(device)
                noise_interp = batch_interpolation_gpu(noise.cpu().numpy())  # NumPy로 변환 후 GPU 보간
                noisy_spectra = clean_spectra + noise_interp
                print(f"Data prep took {time.time() - t1:.2f}s", flush=True)

                t2 = time.time()
                input_dct = process_batch_dct_gpu(noisy_spectra)
                target_dct = process_batch_dct_gpu(noise_interp)
                inp_norm = (input_dct - global_mean) / (global_std + EPS)
                tgt_norm = (target_dct - global_mean) / (global_std + EPS)
                inp_t = inp_norm.reshape(bsz, 1, 1600)
                out_t = tgt_norm.reshape(bsz, 1, 1600)
                print(f"DCT took {time.time() - t2:.2f}s", flush=True)

                t3 = time.time()
                optimizer.zero_grad()
                preds = model(inp_t)
                loss = nn.MSELoss()(preds, out_t)
                loss.backward()
                optimizer.step()
                print(f"Model step took {time.time() - t3:.2f}s", flush=True)

                epoch_train_loss += loss.item()
                global_step += 1
                writer.add_scalar("train_loss", loss.item(), global_step)
                
                del noisy_spectra, noise_interp, input_dct, target_dct
                torch.cuda.empty_cache()
                print(f"Batch {idx+1} total: {time.time() - t0:.2f}s", flush=True)
            except Exception as e:
                print(f"Batch {idx+1} error: {e}", flush=True)
                raise

        epoch_train_loss /= len(train_loader)
        print(f"[Epoch {epoch}] Train Loss: {epoch_train_loss:.6f}, Time: {time.time() - epoch_start:.2f}s", flush=True)

        # 검증
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for noise_v in valid_loader:
                noise_v = noise_v.squeeze().to(device)
                bsz_v = noise_v.shape[0]
                clean_v = torch.from_numpy(gen_train.generator(1600, bsz_v).T).to(device)
                noise_interp_v = batch_interpolation_gpu(noise_v.cpu().numpy())
                noisy_v = clean_v + noise_interp_v
                inp_dct = process_batch_dct_gpu(noisy_v)
                tgt_dct = process_batch_dct_gpu(noise_interp_v)
                inp_norm_v = (inp_dct - global_mean) / (global_std + EPS)
                tgt_norm_v = (tgt_dct - global_mean) / (global_std + EPS)
                inp_v = inp_norm_v.reshape(bsz_v, 1, 1600)
                out_v = tgt_norm_v.reshape(bsz_v, 1, 1600)
                preds_v = model(inp_v)
                v_loss = nn.MSELoss()(preds_v, out_v)
                valid_loss += v_loss.item()
        valid_loss /= len(valid_loader)
        writer.add_scalar("valid_loss", valid_loss, global_step)
        scheduler.step(valid_loss)
        print(f"[Epoch {epoch}] Global Step: {global_step} | Train Loss: {epoch_train_loss:.6f} | Valid Loss: {valid_loss:.6f}", flush=True)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            ckpt_path = os.path.join(save_model_dir(config), "best_model.pt")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_step': global_step,
                'epoch': epoch,
                'loss': valid_loss
            }, ckpt_path)
            print(f"Saved best model at {ckpt_path}, Valid Loss: {valid_loss:.6f}", flush=True)

    writer.close()
    print(f"Training finished. Best valid loss: {best_valid_loss:.4f}", flush=True)

###############################################################################
# Batch Prediction Pipeline
###############################################################################

def batch_predict(config):
    logger.info("Starting batch prediction pipeline...")
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    merged_mat = os.path.join(config.batch_predict_root, f"noisy_{script_name}.mat")
    out_mat = merge_txt_to_single_key_mat(config.raw_predict_base, merged_mat, 1600, 1280)
    if not out_mat or not os.path.exists(out_mat):
        logger.error("Merge txt to mat failed")
        return

    raw_data_matrix = loadmat(out_mat)["data_matrix"].astype(np.float64)
    logger.info(f"Raw data_matrix shape: {raw_data_matrix.shape}")

    # Wavelet 선택적 적용
    processed_matrix = raw_data_matrix.copy()
    if config.wavelet_model and config.wavelet_level:
        for i in range(processed_matrix.shape[0]):
            baseline = wavelet_baseline(processed_matrix[i, :], config.wavelet_model, config.wavelet_level)
            processed_matrix[i, :] -= baseline
        logger.info("Wavelet baseline correction applied")

    # SVD 배경 제거
    U, s, Vt = svd(processed_matrix, full_matrices=False)
    s_mod = s.copy()
    s_mod[0] *= config.fade_factor
    data_bg_removed = U @ np.diag(s_mod) @ Vt
    logger.info(f"After SVD BG removal: {data_bg_removed.shape}")

    # 훈련 데이터에서 글로벌 통계 로드
    train_noise_mat = os.path.join(config.train_data_root, f"noise_{script_name}.mat")
    if not os.path.exists(train_noise_mat):
        logger.error(f"Noise data file not found: {train_noise_mat}")
        return
    train_dict = loadmat(train_noise_mat)
    global_mean, global_std = float(train_dict['global_mean']), float(train_dict['global_std'])
    logger.info(f"Loaded global_mean={global_mean:.4f}, global_std={global_std:.4f}")

    # 모델 로드
    model_file = os.path.join(config.test_model_dir, f"{config.global_step}.pt")
    if not os.path.exists(model_file):
        logger.error(f"Model file not found: {model_file}")
        return
    state = torch.load(model_file, map_location='cpu')
    model = eval(f"{config.model_name}(1,1)")
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    model.eval()
    device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Prediction model loaded on {device}")

    # 배치 예측
    predicted_noise = np.zeros_like(data_bg_removed)
    inp_dct = process_batch_dct(data_bg_removed)
    inp_norm = (inp_dct - global_mean) / (global_std + EPS)
    inp_t = torch.from_numpy(inp_norm.reshape(-1, 1, 1600).astype(np.float32)).to(device)
    with torch.no_grad():
        with autocast('cuda' if config.use_gpu else 'cpu'):
            pred_out = model(inp_t).cpu().numpy().reshape(-1, 1600)
    pred_trans = pred_out * (global_std + EPS) + global_mean
    predicted_noise = process_batch_idct(pred_trans)
    denoised = raw_data_matrix - predicted_noise
    logger.info(f"Predicted noise shape: {predicted_noise.shape}, Denoised shape: {denoised.shape}")

    # 시각화
    sample_idx = 0
    row_spec = data_bg_removed[sample_idx, :]
    row_dct = dct(row_spec, type=2, norm='ortho')
    row_norm = (row_dct - global_mean) / (global_std + EPS)
    inp_t = torch.from_numpy(row_norm.reshape(1, 1, -1).astype(np.float32)).to(device)
    with torch.no_grad():
        with autocast('cuda' if config.use_gpu else 'cpu'):
            pred_out = model(inp_t).cpu().numpy().reshape(-1)
    pred_trans = pred_out * (global_std + EPS) + global_mean
    row_noise = idct(pred_trans, type=2, norm='ortho')
    denoised_sample = raw_data_matrix[sample_idx, :] - row_noise

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()
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

    plt.figure("BatchPredict: Comparison")
    plt.plot(raw_data_matrix[0], label="Raw", color='cyan')
    plt.plot(predicted_noise[0], label="Predicted Noise", color='goldenrod')
    plt.plot(denoised[0], label="Denoised", color='magenta')
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.title("Sample Comparison (Raw, Predicted Noise, Denoised)")
    plt.legend()
    plt.show()

    # 결과 저장
    out_dir = test_result_dir(config)
    out_name = os.path.join(out_dir, f"{script_name}.mat")
    savemat(out_name, {"raw_spectra": raw_data_matrix, "predicted_noise": predicted_noise, "denoised": denoised})
    logger.info(f"Saved results to {out_name}")

###############################################################################
# Main Function
###############################################################################

def main(config):
    print("Starting main function...")
    check_dir(config)
    if config.is_training:
        train(config)
    if config.is_batch_predicting:
        batch_predict(config)

if __name__ == "__main__":
    print("Script started. Ensure TensorFlow processes are terminated to avoid GPU conflicts.")
    opt = DefaultConfig()
    opt.visualize_steps = True
    main(opt)
    plt.show()