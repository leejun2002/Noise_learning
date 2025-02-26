import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial

import pywt
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'

from scipy.fftpack import dct, idct
from scipy.interpolate import interp1d
from scipy.linalg import svd

from multiprocessing import Pool
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve

# Mixed Precision
from torch.cuda.amp import autocast, GradScaler

# 기존 사용자 모듈들
from Model import UNet_CA, UNet, UNet_CA6, UNet_CA7
from Spectra_generator import spectra_generator
from Make_dataset import Read_data, Make_dataset
from config import DefaultConfig

EPS = 1e-6

###############################################################################
# 0. 헬퍼 함수들 (동일 파일 내에 모듈화)
###############################################################################

def wavelet_baseline(y, wavelet=None, level=None):
    """
    Wavelet decomposition을 사용한 baseline 추정.
    detail 계수를 모두 0으로 만들어 low-frequency 성분만 복원한 뒤,
    원본에서 뺄 수 있도록 반환.
    """
    coeffs = pywt.wavedec(y, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    baseline = pywt.waverec(coeffs, wavelet)
    return baseline[:len(y)]

def length_interpolation(fpath, row_points=1600, zero_cut=0.0):
    """
    단일 txt 파일(x, y) 로드 후 row_points 길이로 보간. 
    zero_cut 이하의 x는 제거.
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
        print(f"[Error] Interpolation in {fpath}: {e}", flush=True)
        return None

def volume_interpolation(all_spectra, target_files=1280):
    """
    여러 txt 파일(N행 x L열) -> target_files행 x L열로 세로축 보간
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
    base_dir 안의 모든 txt를 (row_points 길이)로 보간하여 stack,
    그 스택을 다시 target_files 높이로 보간해 최종 (target_files x row_points) 
    행렬 data_matrix로 만든 뒤 .mat 저장.
    """
    file_list = sorted(glob.glob(os.path.join(base_dir, '**', '*.txt'), recursive=True))
    if not file_list:
        print(f"[오류] '{base_dir}' 내 .txt 없음.")
        return None

    # 병렬 row 보간
    with Pool() as pool:
        worker_func = partial(length_interpolation, row_points=row_points, zero_cut=0.0)
        results = pool.map(worker_func, file_list)
    results = [r for r in results if r is not None]
    if not results:
        print("[오류] 유효한 스펙트럼이 없습니다.", flush=True)
        return None

    all_spectra = np.array(results, dtype=np.float64)
    print(f"[merge_txt] shape from files: {all_spectra.shape}", flush=True)

    # 세로축 보간
    data_matrix = volume_interpolation(all_spectra, target_files=target_files)
    print(f"[merge_txt] final shape: {data_matrix.shape}", flush=True)

    sio.savemat(out_mat, {"data_matrix": data_matrix})
    print(f"[merge_txt] saved => {out_mat}", flush=True)
    return out_mat

def apply_wavelet(data, wavelet_model, wavelet_level):
    """
    여러 스펙트럼(N x L) 각각에 wavelet baseline 제거.
    """
    N = data.shape[0]
    out = np.zeros_like(data)
    for i in range(N):
        baseline = wavelet_baseline(data[i, :], wavelet_model, wavelet_level)
        out[i, :] = data[i, :] - baseline
    return out

def apply_svd(data, remove_svs=1, fade_factor=1.0):
    """
    data(N x L)에 SVD 후, 상위 remove_svs개의 특잇값에 fade_factor 적용.
    """
    U, s, Vt = svd(data, full_matrices=False)
    s_mod = s.copy()
    for i in range(remove_svs):
        if i < len(s_mod):
            s_mod[i] *= fade_factor
    data_svd = U @ np.diag(s_mod) @ Vt
    return data_svd

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
# 1. train
###############################################################################
def train(config):
    """
    1) 여러 txt -> row interpolation -> (N x L)
    2) volume interpolation -> (target_files x L)
    3) wavelet 제거
    4) SVD 제거
    5) noise mat 저장 + DCT global mean/std 계산
    6) 모델 학습
    """
    row_points = 1600
    target_files = 1280
    zero_cut = 0.0

    # 1) raw_noise_base 폴더 안 txt들
    base_folder = config.raw_noise_base
    subfolders = ["1", "2", "3"]
    txt_files = []
    for sf in subfolders:
        sf_path = os.path.join(base_folder, sf)
        fs = glob.glob(os.path.join(sf_path, "*.txt"))
        txt_files.extend(fs)
    if not txt_files:
        print("[오류] 학습용 txt 파일 없음.")
        return

    # row interpolation 병렬
    with Pool() as pool:
        func = partial(length_interpolation, row_points=row_points, zero_cut=zero_cut)
        results = pool.map(func, txt_files)
    results = [r for r in results if r is not None]
    if not results:
        print("[오류] row interpolation 실패")
        return

    data = np.array(results, dtype=np.float64)
    # volume interpolation (세로축 보간)
    data = volume_interpolation(data, target_files=target_files)

    # wavelet
    for i in range(data.shape[0]):
        baseline = wavelet_baseline(data[i, :], config.wavelet_model, config.wavelet_level)
        data[i, :] = data[i, :] - baseline

    # SVD 배경 제거 (모델 학습 시점에 사용한다고 가정)
    U, s, Vt = svd(data, full_matrices=False)
    s_mod = s.copy()
    remove_svs = 1
    if remove_svs <= len(s_mod):
        s_mod[0] *= config.fade_factor
    noise_processed = U @ np.diag(s_mod) @ Vt

    # noise .mat + DCT 통계
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    out_mat = os.path.join(config.train_data_root, f"noisE_{script_name}.mat")
    sio.savemat(out_mat, {"noise": noise_processed})
    print(f"[train] saved noise => {out_mat}")

    # global mean/std (DCT)
    noise_dct = np.zeros_like(noise_processed)
    for i in range(noise_processed.shape[0]):
        noise_dct[i, :] = dct(noise_processed[i, :], type=2, norm='ortho')
    global_mean = noise_dct.mean()
    global_std = noise_dct.std()

    # 압축하여 저장
    sio.savemat(out_mat, {
        "noise": noise_processed,
        "global_mean": global_mean,
        "global_std": global_std
    }, do_compression=True)
    print(f"[train] global_mean={global_mean:.4f}, global_std={global_std:.4f}")

    # ===================== 모델 학습 파트 =====================
    # (이 부분은 기존 코드를 그대로 옮김)
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from Spectra_generator import spectra_generator
    from Make_dataset import Read_data, Make_dataset

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = eval(f"{config.model_name}(1,1)")
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

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
            noise = noise.squeeze().numpy()
            bsz, spec_len = noise.shape

            # clean data
            clean = gen_train.generator(spec_len, bsz).T  # shape=(bsz, spec_len)
            noisy = clean + noise

            # DCT & norm
            inp_dct = np.zeros_like(noisy)
            tgt_dct = np.zeros_like(noise)
            for j in range(bsz):
                inp_dct[j, :] = dct(noisy[j, :], type=2, norm='ortho')
                tgt_dct[j, :] = dct(noise[j, :], type=2, norm='ortho')
            inp_norm = (inp_dct - global_mean) / (global_std + EPS)
            tgt_norm = (tgt_dct - global_mean) / (global_std + EPS)

            inp_norm = inp_norm.reshape(bsz, 1, spec_len).astype(np.float32)
            tgt_norm = tgt_norm.reshape(bsz, 1, spec_len).astype(np.float32)

            inp_t = torch.from_numpy(inp_norm).to(device)
            tgt_t = torch.from_numpy(tgt_norm).to(device)

            preds = model(inp_t)
            loss = nn.MSELoss()(preds, tgt_t)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            global_step += 1
            writer.add_scalar("train_loss", loss.item(), global_step)

        epoch_train_loss /= len(train_loader)

        # validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for noise_v in valid_loader:
                noise_v = noise_v.squeeze().numpy()
                bsz, spec_len = noise_v.shape
                clean_v = gen_valid.generator(spec_len, bsz).T
                noisy_v = clean_v + noise_v

                in_dct = np.zeros_like(noisy_v)
                tg_dct = np.zeros_like(noise_v)
                for j in range(bsz):
                    in_dct[j, :] = dct(noisy_v[j, :], type=2, norm='ortho')
                    tg_dct[j, :] = dct(noise_v[j, :], type=2, norm='ortho')
                in_norm = (in_dct - global_mean) / (global_std + EPS)
                tg_norm = (tg_dct - global_mean) / (global_std + EPS)

                in_norm = in_norm.reshape(bsz,1,spec_len).astype(np.float32)
                tg_norm = tg_norm.reshape(bsz,1,spec_len).astype(np.float32)
                in_t = torch.from_numpy(in_norm).to(device)
                tg_t = torch.from_numpy(tg_norm).to(device)

                pred_v = model(in_t)
                vloss = nn.MSELoss()(pred_v, tg_t)
                valid_loss += vloss.item()

        valid_loss /= len(valid_loader)
        writer.add_scalar("valid_loss", valid_loss, global_step)
        scheduler.step(valid_loss)

        print(f"[Epoch {epoch}] Step={global_step}  train_loss={epoch_train_loss:.6f}, valid_loss={valid_loss:.6f}")
        # 예시로 50 epoch마다 ckpt 저장
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

    writer.close()
    print(f"[train] Finished. Best valid loss = {best_valid_loss:.4f}", flush=True)

###############################################################################
# 2. predict
###############################################################################
def predict(config, row_points=1600, zero_cut=0.0):
    """
    SVD를 쓸지 여부(config.use_svd) 에 따라:
      - wavelet baseline 제거
      - (옵션) SVD 제거
      - DCT & global_mean/std 정규화
      - 모델 추론
      - denoised = raw - predicted_noise
    단일 txt 파일을 각각 처리.
    """
    txt_files = sorted(glob.glob(os.path.join(config.predict_root, '*.txt')))
    if not txt_files:
        print(f"[predict] No .txt in {config.predict_root}")
        return

    # global mean/std
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    train_noise_mat = os.path.join(config.train_data_root, f"noisE_{script_name}.mat")
    if not os.path.exists(train_noise_mat):
        print("[predict] no train noise mat found:", train_noise_mat)
        return

    train_dict = sio.loadmat(train_noise_mat, struct_as_record=False, squeeze_me=True)
    if 'global_mean' not in train_dict or 'global_std' not in train_dict:
        print("[predict] global_mean/std not found.")
        return

    global_mean = float(train_dict['global_mean'])
    global_std  = float(train_dict['global_std'])

    # 모델 로드
    model_file = os.path.join(config.test_model_dir, f"{config.global_step}.pt")
    if not os.path.exists(model_file):
        print("[predict] model file not found:", model_file)
        return
    state = torch.load(model_file, map_location='cpu')
    model = eval(f"{config.model_name}(1,1)")
    model.load_state_dict({k.replace('module.', ''): v for k,v in state['model'].items()})
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    print(f"[predict] loaded model from {model_file}")
    print(f"[predict] global_mean={global_mean:.4f}, global_std={global_std:.4f}")
    print(f"[predict] use_svd={config.use_svd} (if True, apply SVD)")

    for fpath in txt_files:
        print(f"\n--- predict file: {fpath}")
        spec = length_interpolation(fpath, row_points=row_points, zero_cut=zero_cut)
        if spec is None:
            print("[predict] interpolation fail or invalid data.")
            continue

        # wavelet
        baseline = wavelet_baseline(spec, config.wavelet_model, config.wavelet_level)
        wv_corrected = spec - baseline
        data_for_pred = wv_corrected.reshape(1,-1)  # shape=(1,L)

        # (옵션) SVD
        if config.use_svd:
            data_for_pred = apply_svd(data_for_pred, remove_svs=1, fade_factor=config.fade_factor)

        # DCT & model inference
        row = data_for_pred[0, :]
        row_dct = dct(row, type=2, norm='ortho')
        row_norm = (row_dct - global_mean) / (global_std + EPS)
        inp_t = torch.from_numpy(row_norm[None, None, :].astype(np.float32))
        if torch.cuda.is_available():
            inp_t = inp_t.cuda()
        with torch.no_grad():
            pred_out = model(inp_t).cpu().numpy().reshape(-1)
        pred_trans = pred_out * (global_std + EPS) + global_mean
        predicted_noise = idct(pred_trans, type=2, norm='ortho')

        # denoised = raw - noise
        denoised = spec - predicted_noise

        # 시각화 (옵션)
        if config.visualize:
            fig, axs = plt.subplots(2,2, figsize=(12,8))
            axs = axs.flatten()

            axs[0].plot(spec, label="Raw", color='blue')
            axs[0].plot(baseline, label="Wavelet Baseline", color='orange')
            axs[0].plot(wv_corrected, label="Wavelet Corrected", color='green')
            axs[0].set_title("Wavelet Correction")
            axs[0].legend()

            axs[1].plot(row_dct, label="DCT", color='purple')
            axs[1].plot(row_norm, label="Normalized", color='red')
            axs[1].set_title("DCT & Normalized")
            axs[1].legend()

            axs[2].plot(predicted_noise, label="Pred Noise", color='red')
            axs[2].set_title("Predicted Noise")
            axs[2].legend()

            axs[3].plot(spec, label="Raw", color='blue')
            axs[3].plot(predicted_noise, label="Noise", color='orange')
            axs[3].plot(denoised, label="Denoised", color='green')
            axs[3].set_title("Denoised")
            axs[3].legend()

            plt.suptitle(f"Predict => {os.path.basename(fpath)}", fontsize=14)
            plt.tight_layout()
            plt.show()

        # save .mat
        fname_noext = os.path.splitext(os.path.basename(fpath))[0]
        out_mat = os.path.join(config.predict_save_root, f"predict_{fname_noext}.mat")
        out_dict = {
            "original_interpolated": spec,
            "wavelet_corrected": wv_corrected,
            "predicted_noise": predicted_noise,
            "denoised": denoised
        }
        sio.savemat(out_mat, out_dict, do_compression=True)
        print(f"[predict] saved => {out_mat}")

    print(f"[predict] all done. total={len(txt_files)} files.")


###############################################################################
# 3. batch_predict
###############################################################################
def batch_predict(config):
    """
    여러 txt -> 하나의 data_matrix (target_files x row_points)
    -> wavelet -> (옵션) SVD -> model -> denoised
    """
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
        print("[batch_predict] merge_txt_to_single_key_mat_1280 failed.")
        return

    tmp = sio.loadmat(out_mat, struct_as_record=False, squeeze_me=True)
    if 'data_matrix' not in tmp:
        print("[batch_predict] data_matrix not found in mat.")
        return

    raw_data_matrix = tmp['data_matrix'].astype(np.float64)
    N, L = raw_data_matrix.shape
    print(f"[batch_predict] raw_data_matrix shape={raw_data_matrix.shape}")

    # wavelet
    processed_matrix = raw_data_matrix.copy()
    processed_matrix = apply_wavelet(processed_matrix, config.wavelet_model, config.wavelet_level)

    # (옵션) SVD
    if config.use_svd:
        processed_matrix = apply_svd(processed_matrix, remove_svs=1, fade_factor=config.fade_factor)

    # load global mean/std
    train_noise_mat = os.path.join(config.train_data_root, f"noisE_{script_name}.mat")
    if not os.path.exists(train_noise_mat):
        print("[batch_predict] train_noise_mat not found")
        return
    train_dict = sio.loadmat(train_noise_mat, struct_as_record=False, squeeze_me=True)
    if 'global_mean' not in train_dict or 'global_std' not in train_dict:
        print("[batch_predict] global_mean/std not found.")
        return
    global_mean = float(train_dict['global_mean'])
    global_std = float(train_dict['global_std'])

    # load model
    model_file = os.path.join(config.test_model_dir, f"{config.global_step}.pt")
    if not os.path.exists(model_file):
        print("[batch_predict] model file not found:", model_file)
        return

    state = torch.load(model_file, map_location='cpu')
    model = eval(f"{config.model_name}(1,1)")
    model.load_state_dict({k.replace('module.', ''): v for k,v in state['model'].items()})
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # inference
    predicted_noise = np.zeros_like(processed_matrix)
    for i in range(N):
        row_spec = processed_matrix[i, :]
        row_dct = dct(row_spec, type=2, norm='ortho')
        row_norm = (row_dct - global_mean) / (global_std + EPS)

        inp_t = torch.from_numpy(row_norm.reshape(1,1,-1).astype(np.float32))
        if torch.cuda.is_available():
            inp_t = inp_t.cuda()

        with torch.no_grad():
            pred_out = model(inp_t).cpu().numpy().reshape(-1)

        pred_trans = pred_out * (global_std + EPS) + global_mean
        row_noise = idct(pred_trans, type=2, norm='ortho')
        predicted_noise[i, :] = row_noise

    denoised = raw_data_matrix - predicted_noise
    print(f"[batch_predict] => predicted_noise shape={predicted_noise.shape}, denoised={denoised.shape}")

    # 시각화 예시(옵션)
    if config.visualize:
        sample_idx = 0
        fig, axs = plt.subplots(2,2, figsize=(12,8))
        axs = axs.flatten()

        # DCT
        row_dct = dct(processed_matrix[sample_idx], type=2, norm='ortho')
        row_norm = (row_dct - global_mean) / (global_std + EPS)

        axs[0].plot(row_dct, color='blue')
        axs[0].set_title("DCT Coeff")

        axs[1].plot(row_norm, color='red')
        axs[1].set_title("Normalized DCT")

        axs[2].plot(predicted_noise[sample_idx], label="Predicted Noise", color='orange')
        axs[2].set_title("Predicted Noise")

        axs[3].plot(denoised[sample_idx], label="Denoised", color='green')
        axs[3].set_title("Denoised Spectrum")

        plt.suptitle("BatchPredict sample idx=0")
        plt.tight_layout()
        plt.show()

    # save mat
    out_dir = config.batch_save_root
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, f"{script_name}.mat")
    out_dict = {
        "raw_spectra": raw_data_matrix,
        "predicted_noise": predicted_noise,
        "denoised": denoised
    }
    sio.savemat(out_path, out_dict, do_compression=True)
    print(f"[batch_predict] saved => {out_path}")


###############################################################################
# 4. main
###############################################################################
def main(config):
    """
    하나의 파일에서 train, predict, batch_predict를 모두 가능하게 하는 구조.
    config에서 on/off를 결정.
    """
    # 폴더 생성
    for d in [config.checkpoint, config.logs, config.test_dir, config.predict_save_root, config.batch_save_root]:
        if not os.path.exists(d):
            os.makedirs(d)

    if config.is_training:
        train(config)
    if config.is_predicting:
        predict(config)
    if config.is_batch_predicting:
        batch_predict(config)


if __name__ == '__main__':
    opt = DefaultConfig()
    # 필요하다면 여기서 opt.use_svd = True / opt.visualize = True 등 설정
    # 예: opt.use_svd = False
    #     opt.visualize = False
    main(opt)