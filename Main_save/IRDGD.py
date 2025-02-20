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
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'

from multiprocessing import Pool
from scipy.interpolate import interp1d
from scipy.linalg import svd

# Mixed Precision (원하면 사용)
from torch.cuda.amp import autocast, GradScaler

# 모델 파일들 (예: U-Net 계열)
from Model import UNet_CA, UNet, UNet_CA6, UNet_CA7
from Spectra_generator import spectra_generator
from Make_dataset import Read_data, Make_dataset
from config import DefaultConfig

EPS = 1e-6

###############################################################################
# RPCA 함수 (Inexact ALM 방식)
###############################################################################
def rpca(M, lam=None, tol=1e-7, max_iter=1000):
    """
    RPCA via Inexact Augmented Lagrange Multiplier (IALM)
    M: 입력 데이터 행렬
    lam: sparsity 파라미터 (default: 1/sqrt(max(m,n)))
    tol: 수렴 허용 오차
    max_iter: 최대 반복 횟수
    반환: L, S (L: 저랭크 성분, S: 희소 성분)
    """
    m, n = M.shape
    norm_M = np.linalg.norm(M, ord='fro')
    if lam is None:
        lam = 1/np.sqrt(max(m, n))
    # 초기화
    L = np.zeros_like(M)
    S = np.zeros_like(M)
    Y = M / max(np.linalg.norm(M, ord=2), np.linalg.norm(M, ord=np.inf)/lam)
    mu = 1.25 / np.linalg.norm(M, ord=2)
    mu_bar = mu * 1e7
    rho = 1.5
    iter = 0
    while iter < max_iter:
        iter += 1
        # Singular Value Thresholding
        U, sigma, Vt = svd(M - S + (1/mu)*Y, full_matrices=False)
        sigma_shrink = np.maximum(sigma - 1/mu, 0)
        L_new = U @ np.diag(sigma_shrink) @ Vt
        
        # Shrinkage for sparse component
        temp = M - L_new + (1/mu)*Y
        S_new = np.sign(temp) * np.maximum(np.abs(temp) - lam/mu, 0)
        
        # 업데이트
        Z = M - L_new - S_new
        Y = Y + mu * Z
        err = np.linalg.norm(Z, ord='fro') / norm_M
        if err < tol:
            break
        mu = min(mu * rho, mu_bar)
        L = L_new
        S = S_new
    return L_new, S_new

###############################################################################
# Global helper functions
###############################################################################
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
    data_matrix = volume_interpolation(all_spectra, target_files=target_files)
    print(f"[merge_txt] Final data_matrix shape: {data_matrix.shape} (target_files x {row_points})", flush=True)
    sio.savemat(out_mat, {"data_matrix": data_matrix})
    print(f"[merge_txt] Saved data_matrix to '{out_mat}'", flush=True)
    return out_mat

###############################################################################
# train pipeline
###############################################################################
def train(config):
    """
    Train 과정:
      - Au film 데이터(txt)를 읽어 보간하여 data_matrix 생성
      - RPCA를 통해 전체 데이터 행렬 M에 대해 M = L + S로 분해하여,
        L을 baseline, S를 residual noise로 사용.
      - S (residual noise)에 대해 DCT 통계를 계산하여 global_mean, global_std를 산출하고,
        해당 정규화 파라미터를 저장한 후 모델 학습에 사용함.
    """
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
    all_spectra = volume_interpolation(all_spectra, target_files=target_files)
    print(f"[train pipeline] After volume interpolation: {all_spectra.shape}", flush=True)

    # RPCA를 통해 M = L + S 분해 (L: baseline, S: residual noise)
    # lam, tol, max_iter는 config 또는 기본값 사용
    lam = config.rpca_lam if hasattr(config, 'rpca_lam') else 1/np.sqrt(max(all_spectra.shape))
    tol = config.rpca_tol if hasattr(config, 'rpca_tol') else 1e-7
    max_iter = config.rpca_max_iter if hasattr(config, 'rpca_max_iter') else 1000
    L_component, S_component = rpca(all_spectra, lam=lam, tol=tol, max_iter=max_iter)
    # S_component를 residual noise로 사용
    noise_processed = S_component
    print(f"[train pipeline] RPCA completed: L shape {L_component.shape}, S shape {S_component.shape}", flush=True)

    # RPCA 전후 시각화 (첫 번째 스펙트럼)
    plt.figure("RPCA Comparison (Train Data)")
    plt.subplot(1,3,1)
    plt.plot(all_spectra[0], label="Original")
    plt.title("Original")
    plt.legend()
    plt.subplot(1,3,2)
    plt.plot(L_component[0], label="Estimated Baseline (L)")
    plt.title("Estimated Baseline")
    plt.legend()
    plt.subplot(1,3,3)
    plt.plot(S_component[0], label="Residual (S)")
    plt.title("Residual (Noise)")
    plt.legend()
    plt.show()

    # DCT를 통해 전역 통계 계산 (Global Z-score normalization) on noise_processed
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    output_mat = os.path.join(config.train_data_root, f"noisE_{script_name}.mat")
    noise_dct = np.zeros_like(noise_processed)
    for i in range(noise_processed.shape[0]):
        noise_dct[i, :] = dct(noise_processed[i, :], type=2, norm='ortho')
    global_mean = np.mean(noise_dct)
    global_std = np.std(noise_dct)

    sio.savemat(output_mat, {
        "noise": noise_processed,
        "global_mean": global_mean,
        "global_std": global_std
    }, do_compression=True)
    print(f"[train pipeline] Noise data saved to {output_mat}")
    print(f"[train pipeline] global_mean={global_mean:.4f}, global_std={global_std:.4f}")

    # 모델 학습 (이하 기존 코드와 동일)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and config.use_gpu) else "cpu")
    model = eval(f"{config.model_name}(1,1)")
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    writer = SummaryWriter(config.logs)
    global_step = 0
    best_valid_loss = 1e9

    from Make_dataset import Read_data, Make_dataset
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
            bsz, spec = noise.shape

            clean_spectra = gen_train.generator(spec, bsz).T
            noisy_spectra = clean_spectra + noise

            input_dct = np.zeros_like(noisy_spectra)
            target_dct = np.zeros_like(noise)
            for j in range(bsz):
                input_dct[j, :] = dct(noisy_spectra[j, :], type=2, norm='ortho')
                target_dct[j, :] = dct(noise[j, :], type=2, norm='ortho')

            input_norm = (input_dct - global_mean) / (global_std + EPS)
            target_norm = (target_dct - global_mean) / (global_std + EPS)

            inp_t = torch.from_numpy(input_norm.reshape(bsz, 1, spec).astype(np.float32)).to(device)
            out_t = torch.from_numpy(target_norm.reshape(bsz, 1, spec).astype(np.float32)).to(device)

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
        print(f"[Epoch {epoch}] Train Loss: {epoch_train_loss:.6f}", flush=True)

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

                inp_v = torch.from_numpy(inp_norm_v.reshape(bsz, 1, spec).astype(np.float32)).to(device)
                out_v = torch.from_numpy(tgt_norm_v.reshape(bsz, 1, spec).astype(np.float32)).to(device)
                preds_v = model(inp_v)
                v_loss = nn.MSELoss()(preds_v, out_v)
                valid_loss += v_loss.item()

        valid_loss /= len(valid_loader)
        writer.add_scalar("valid_loss", valid_loss, global_step)
        scheduler.step(valid_loss)
        print(f"[Epoch {epoch}] Global Step: {global_step} | Valid Loss: {valid_loss:.6f}", flush=True)

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

    print(f"[train] Finished. Best valid loss = {best_valid_loss:.4f}")

###############################################################################
# batch_predict
###############################################################################
def batch_predict(config):
    """
    Batch_predict 과정:
      - Test 데이터(txt)를 읽어 보간하여 raw_data_matrix 생성
      - 전체 테스트 데이터의 DCT 통계를 계산하여 global_mean, global_std를 업데이트 (Domain Adaptation)
      - 해당 정규화 파라미터를 이용해 모델 예측 후 역 DCT로 noise를 추출하고,
        원본에서 noise를 빼 denoised 결과를 계산.
    """
    import os
    import sys
    import glob
    import numpy as np
    import torch
    import scipy.io as sio
    from scipy.fftpack import dct, idct
    import matplotlib.pyplot as plt

    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    row_points = 1600
    target_files = 1280

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
        print("[오류] data_matrix key not found in mat", flush=True)
        return
    raw_data_matrix = tmp['data_matrix'].astype(np.float64)
    print(f"[batch_predict] raw_data_matrix shape = {raw_data_matrix.shape}", flush=True)

    # 학습 시 저장한 정규화 파라미터 (참고용)
    train_noise_mat = os.path.join(config.train_data_root, f"noisE_{script_name}.mat")
    if not os.path.exists(train_noise_mat):
        print("[오류] noise_data.mat not found", flush=True)
        return
    train_dict = sio.loadmat(train_noise_mat, struct_as_record=False, squeeze_me=True)
    if 'global_mean' not in train_dict or 'global_std' not in train_dict:
        print("[오류] global_mean/std not found in noise_data.mat", flush=True)
        return
    old_global_mean = float(train_dict['global_mean'])
    old_global_std = float(train_dict['global_std'])
    print(f"[batch_predict] Loaded train global_mean={old_global_mean}, global_std={old_global_std}")

    # Domain Adaptation: 전체 테스트 데이터의 DCT 통계를 계산하여 업데이트
    test_dct = dct(raw_data_matrix, type=2, norm='ortho', axis=1)
    new_global_mean = np.mean(test_dct)
    new_global_std = np.std(test_dct)
    print(f"[Domain Adaptation] Updating normalization parameters:")
    print(f"  Old (Train): mean={old_global_mean:.4f}, std={old_global_std:.4f}")
    print(f"  New (Test):  mean={new_global_mean:.4f}, std={new_global_std:.4f}")
    global_mean = new_global_mean
    global_std = new_global_std

    # 모델 로드
    model_file = os.path.join(config.test_model_dir, f"{config.global_step}.pt")
    if not os.path.exists(model_file):
        print(f"[오류] Model file not found: {model_file}", flush=True)
        return

    state = torch.load(model_file, map_location='cpu')
    model = eval(f"{config.model_name}(1,1)")
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    model.eval()
    if torch.cuda.is_available() and config.use_gpu:
        model = model.cuda()

    # raw 데이터에 대해 DCT 변환 → 정규화 → 모델 예측 → 역 DCT 적용
    N, L = raw_data_matrix.shape
    predicted_noise = np.zeros_like(raw_data_matrix)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and config.use_gpu) else "cpu")
    for i in range(N):
        spec = raw_data_matrix[i, :]
        spec_dct = dct(spec, type=2, norm='ortho')
        spec_norm = (spec_dct - global_mean) / (global_std + EPS)

        inp_t = torch.from_numpy(spec_norm.reshape(1, 1, L).astype(np.float32)).to(device)
        with torch.no_grad():
            pred_out = model(inp_t).cpu().numpy().reshape(-1)
        pred_trans = pred_out * (global_std + EPS) + global_mean
        spec_noise = idct(pred_trans, type=2, norm='ortho')
        predicted_noise[i, :] = spec_noise

    denoised = raw_data_matrix - predicted_noise

    plt.figure("BatchPredict: Sample")
    plt.plot(raw_data_matrix[0], label="Raw")
    plt.plot(predicted_noise[0], label="Predicted Noise")
    plt.plot(denoised[0], label="Denoised")
    plt.legend()
    plt.title("Sample spectrum result")
    plt.show()

    out_dir = config.batch_save_root
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = os.path.join(out_dir, f"{script_name}.mat")
    sio.savemat(out_name, {
        "raw_spectra": raw_data_matrix,
        "predicted_noise": predicted_noise,
        "denoised": denoised
    }, do_compression=True)
    print(f"[batch_predict] Saved results to {out_name}", flush=True)

###############################################################################
# 경로 및 기타 함수들
###############################################################################
def check_dir(config):
    for d in [config.checkpoint, config.logs, config.test_dir]:
        if not os.path.exists(d):
            os.mkdir(d)

def save_model_dir(config):
    d = os.path.join(config.checkpoint, config.Instrument, config.model_name, "batch_" + str(config.batch_size))
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def test_result_dir(config):
    rdir = os.path.join(config.batch_save_root, config.Instrument, config.model_name, "step_" + str(config.global_step))
    if not os.path.exists(rdir):
        os.makedirs(rdir)
    return rdir

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
    # 예: RPCA 관련 파라미터는 config에 따로 지정 (예: rpca_lam, rpca_tol, rpca_max_iter)
    main(opt)
    plt.show()
