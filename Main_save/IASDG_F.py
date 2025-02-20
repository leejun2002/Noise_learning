import os
import glob
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from scipy.fftpack import dct, idct
from scipy.io import savemat
from scipy.interpolate import interp1d
from scipy.linalg import svd

import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'

from numba import njit
from multiprocessing import Pool

# 모델 파일들 (예: U-Net 계열)
from Model import UNet_CA, UNet, UNet_CA6, UNet_CA7
from Spectra_generator import spectra_generator
from Make_dataset import Read_data, Make_dataset
from config import DefaultConfig

EPS = 1e-6

##########################################
# 1. 전처리 및 데이터 로딩 (병렬 처리 적용)
##########################################

def precompute_DTD(L, lam):
    """
    데이터 길이 L에 대해, 행 방향(axis=0)에서 2차 차분 행렬을 계산한 후,
    D^T*D에 lam을 곱한 결과를 반환합니다.
    (결과 크기는 (L, L)입니다.)
    """
    D = np.diff(np.eye(L), 2, axis=0)  # D.shape: (L-2, L)
    return lam * D.T.dot(D)

@njit
def als_baseline_numba(y, DTD, p, niter):
    """
    Numba JIT를 사용하여 ALS baseline correction을 수행합니다.
    
    Args:
        y (1D np.array): 입력 신호.
        DTD (2D np.array): lam * (D^T D), shape: (L, L)
        p (float): 비대칭성 파라미터.
        niter (int): 반복 횟수.
    
    Returns:
        Z (1D np.array): 추정된 baseline.
    """
    L = y.shape[0]
    w = np.ones(L)
    for it in range(niter):
        A = DTD.copy()
        for j in range(L):
            A[j, j] += w[j]
        Z = np.linalg.solve(A, w * y)
        for j in range(L):
            if y[j] > Z[j]:
                w[j] = p
            else:
                w[j] = 1 - p
    return Z

def als_baseline(y, lam=1e5, p=0.01, niter=10, visualize=False, DTD=None):
    """
    ALS baseline correction.
    
    Args:
        y (1D np.array): 입력 신호.
        lam (float): baseline의 부드러움을 제어.
        p (float): 비대칭성 파라미터.
        niter (int): 반복 횟수.
        visualize (bool): 개별 ALS 내부 진행률 출력 여부.
        DTD (2D np.array, optional): 미리 계산된 lam * (D^T D) (크기 (L, L)).
        
    Returns:
        baseline (1D np.array): 추정된 baseline.
    """
    L = len(y)
    if DTD is None:
        D = np.diff(np.eye(L), 2, axis=0)
        DTD = lam * D.T.dot(D)
    Z = als_baseline_numba(y, DTD, p, niter)
    if visualize:
        print("ALS complete for current row.", flush=True)
    return Z

def process_file(fpath, zero_cut, row_new_length):
    try:
        arr = np.loadtxt(fpath)
    except Exception as e:
        print(f"Error loading {fpath}: {e}", flush=True)
        return None
    if arr.ndim != 2 or arr.shape[1] < 2:
        print(f"Skipping {fpath}, shape={arr.shape}", flush=True)
        return None
    x_raw, y_raw = arr[:, 0], arr[:, 1]
    mask = (x_raw >= zero_cut)
    x_cut = x_raw[mask]
    y_cut = y_raw[mask]
    if x_cut.size < 2:
        print(f"Skipping {fpath}: not enough data", flush=True)
        return None
    x_min, x_max = x_cut.min(), x_cut.max()
    if x_min == x_max:
        return None
    x_new = np.linspace(x_min, x_max, row_new_length)
    try:
        f_intp = interp1d(x_cut, y_cut, kind='cubic')
        y_new = f_intp(x_new)
    except Exception as e:
        print(f"Interpolation error in {fpath}: {e}", flush=True)
        return None
    return y_new

def pipeline_and_save_noise(base_folder, output_mat, config, subfolders=("1", "2", "3"),
                              zero_cut=0.0, row_new_length=1600, col_new_length=640, remove_svs=1):
    """
    pipeline: 텍스트 파일들을 읽어 행/열 보간, ALS baseline correction, SVD 배경 제거까지 수행하고,
    DCT 및 정규화 없이 raw noise (baseline과 BG 제거된 noise)를 .mat 파일에 저장합니다.
    """
    eps = 1e-6
    # Step 1: Row Interpolation (병렬 처리)
    txt_files = []
    for sf in subfolders:
        sf_path = os.path.join(base_folder, sf)
        fpaths = glob.glob(os.path.join(sf_path, "*.txt"))
        txt_files.extend(fpaths)
    if not txt_files:
        print("No txt files found in subfolders:", subfolders, flush=True)
        return None
    txt_files = sorted(txt_files)
    with Pool() as pool:
        results = pool.starmap(process_file, [(fpath, zero_cut, row_new_length) for fpath in txt_files])
    spectra_list = [res for res in results if res is not None]
    if not spectra_list:
        print("No valid spectra found.", flush=True)
        return None
    all_spectra = np.array(spectra_list).T
    print("After row interpolation:", all_spectra.shape, flush=True)
    if config.visualize_steps:
        plt.figure("Row Interpolation")
        plt.imshow(all_spectra, aspect='auto', cmap='jet')
        plt.title("After Row Interpolation")
        plt.colorbar()
        plt.show()
    # Step 2: Column Interpolation
    num_rows, num_cols = all_spectra.shape
    col_idx = np.arange(num_cols)
    col_new = np.linspace(0, num_cols - 1, col_new_length)
    all_spectra_2d = np.zeros((num_rows, col_new_length), dtype=all_spectra.dtype)
    for i in range(num_rows):
        f_c = interp1d(col_idx, all_spectra[i, :], kind='linear')
        all_spectra_2d[i, :] = f_c(col_new)
    print("After column interpolation:", all_spectra_2d.shape, flush=True)
    if config.visualize_steps:
        plt.figure("Column Interpolation")
        plt.imshow(all_spectra_2d, aspect='auto', cmap='jet')
        plt.title("After Column Interpolation")
        plt.colorbar()
        plt.show()
    # Step 3: ALS Baseline Correction (컬럼별)
    data_baseline = np.zeros_like(all_spectra_2d)
    total_cols = all_spectra_2d.shape[1]
    DTD = precompute_DTD(row_new_length, lam=1e5)
    for j in range(total_cols):
        baseline = als_baseline(all_spectra_2d[:, j], lam=1e5, p=0.01, niter=10, visualize=False, DTD=DTD)
        data_baseline[:, j] = baseline
        progress = ((j+1) / total_cols) * 100
        sys.stdout.write(f"\rOverall ALS progress: {progress:.1f}% complete")
        sys.stdout.flush()
    print("", flush=True)
    data_bc_removed = all_spectra_2d - data_baseline
    print("After ALS baseline correction:", data_bc_removed.shape, flush=True)
    if config.visualize_steps:
        plt.figure("Baseline Correction")
        plt.imshow(data_bc_removed, aspect='auto', cmap='jet')
        plt.title("After Baseline Removal")
        plt.colorbar()
        plt.show()
    # Step 4: SVD Background Removal
    U, s, Vt = svd(data_bc_removed, full_matrices=False)
    s_mod = s.copy()
    for i in range(remove_svs):
        if i < len(s_mod):
            s_mod[i] *= config.fade_factor
    data_bg_removed = U @ np.diag(s_mod) @ Vt
    print("After SVD background removal:", data_bg_removed.shape, flush=True)
    if config.visualize_steps:
        plt.figure("SVD BG Removal")
        plt.imshow(data_bg_removed, aspect='auto', cmap='jet')
        plt.title("After SVD BG Removal")
        plt.colorbar()
        plt.show()
    # pipeline에서는 DCT 및 정규화 없이, data_bg_removed (baseline과 BG 제거된 raw noise)를 저장
    print("Saving noise data without DCT and normalization.", flush=True)
    savemat(output_mat, {"noise": data_bg_removed})
    print("\nSaved noise_data.mat =>", output_mat, flush=True)
    return output_mat

##########################################
# 2. 모델 학습 및 최적화 (Train)
##########################################

def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # pipeline에서 생성한 .mat 파일을 로드
    output_mat = os.path.join(config.train_data_root, "noise_data.mat")
    print("[INFO] Start pipeline_and_save_noise ...", flush=True)
    pipeline_and_save_noise(
        base_folder=config.raw_noise_base,
        output_mat=output_mat,
        config=config,
        subfolders=("1", "2", "3"),
        zero_cut=0.0,
        row_new_length=1600,
        col_new_length=640,
        remove_svs=1
    )
    plt.show()
    print("[INFO] Noise data .mat 생성 완료 =>", output_mat, flush=True)
    
    # 여기서 pipeline에서 저장한 raw noise를 불러와서 global z‑score normalization의 mean, std 계산
    noise_data_dict = sio.loadmat(output_mat, struct_as_record=False, squeeze_me=True)
    raw_noise = noise_data_dict['noise']  # 예: (1600, 640)
    
    # (전체 noise에 대한 DCT를 구한 후 평균, std를 계산)
    noise_dct = np.zeros_like(raw_noise)
    for j in range(raw_noise.shape[1]):
        noise_dct[:, j] = dct(raw_noise[:, j], type=2, norm='ortho')
    global_mean = np.mean(noise_dct)
    global_std = np.std(noise_dct)
    print(f"Calculated global_mean: {global_mean:.4f}, global_std: {global_std:.4f}", flush=True)
    
    # 업데이트: 원래 .mat 파일에 global_mean과 global_std도 추가하여 저장
    noise_data_dict.update({"global_mean": global_mean, "global_std": global_std})
    sio.savemat(output_mat, noise_data_dict, do_compression=True)
    print("Updated noise_data.mat with global normalization parameters.", flush=True)
    
    # 모델 생성
    model = eval("{}(1,1)".format(config.model_name))
    print(model, flush=True)
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
        model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=5, verbose=True)
    
    save_model_path = save_model_dir(config)
    save_log = save_log_dir(config)
    writer = SummaryWriter(save_log)
    
    global_step = -1
    best_valid_loss = float('inf')
    
    ### <--- [수정] early stopping 관련 변수 제거
    # patience = 100   # early stopping patience
    # no_improve_epochs = 0
    
    best_models = []  # (valid_loss, epoch, global_step, checkpoint_file)
    
    # Dataloader 준비
    reader = Read_data(config.train_data_root, config.valid_ratio)
    train_set, valid_set = reader.read_file()
    TrainSet, ValidSet = Make_dataset(train_set), Make_dataset(valid_set)
    train_loader = DataLoader(TrainSet, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(ValidSet, batch_size=256, pin_memory=True)
    
    gen_train = spectra_generator()
    gen_valid = spectra_generator()
    
    for epoch in range(config.max_epoch):
        model.train()
        epoch_train_loss = 0.0
        
        for idx, noise in enumerate(train_loader):
            # noise.shape => (batch, spec)
            noise = noise.squeeze().numpy()  # (batch, spec)
            spectra_num, spec = noise.shape
            
            # Clean 스펙트럼 생성
            clean_spectra = gen_train.generator(spec, spectra_num).T  # shape: (spec, spectra_num)
            clean_spectra = clean_spectra  # (spec, spectra_num)
            
            # noisy_spectra = clean + noise
            noisy_spectra = clean_spectra + noise  # shape: (spec, spectra_num)
            # 전치 => (spectra_num, spec)
            noisy_spectra = noisy_spectra
            clean_spectra = clean_spectra
            
            # DCT 적용
            input_coef = np.zeros_like(noisy_spectra)
            target_coef = np.zeros_like(noise)
            for i in range(spectra_num):
                input_coef[i, :] = dct(noisy_spectra[i, :], type=2, norm='ortho')
                target_coef[i, :] = dct(noise[i, :], type=2, norm='ortho')
            
            # 전역 z‑score normalization
            input_norm = (input_coef - global_mean) / (global_std + EPS)
            target_norm = (target_coef - global_mean) / (global_std + EPS)
            
            # 3D 텐서 변환 => (batch, 1, spec)
            input_norm = input_norm.reshape(-1, 1, spec).astype(np.float32)
            target_norm = target_norm.reshape(-1, 1, spec).astype(np.float32)
            
            inp_t = torch.from_numpy(input_norm).to(device)
            out_t = torch.from_numpy(target_norm).to(device)
            
            global_step += 1
            preds = model(inp_t)
            loss = nn.MSELoss()(preds, out_t)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            writer.add_scalar("train loss", loss.item(), global_step)
        
        epoch_train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        valid_loss = 0.0
        for idx_v, noise_v in enumerate(valid_loader):
            noise_v = noise_v.squeeze().numpy()  # (batch, spec)
            spectra_num, spec = noise_v.shape
            clean_spectra = gen_valid.generator(spec, spectra_num).T
            noisy_spectra = clean_spectra + noise_v
            
            input_coef_v = np.zeros_like(noisy_spectra)
            target_coef_v = np.zeros_like(noise_v)
            for i in range(spectra_num):
                input_coef_v[i, :] = dct(noisy_spectra[i, :], type=2, norm='ortho')
                target_coef_v[i, :] = dct(noise_v[i, :], type=2, norm='ortho')
            
            input_norm_v = (input_coef_v - global_mean) / (global_std + EPS)
            target_norm_v = (target_coef_v - global_mean) / (global_std + EPS)
            
            input_norm_v = input_norm_v.reshape(-1, 1, spec).astype(np.float32)
            target_norm_v = target_norm_v.reshape(-1, 1, spec).astype(np.float32)
            
            inp_v = torch.from_numpy(input_norm_v).to(device)
            out_v = torch.from_numpy(target_norm_v).to(device)
            
            preds_v = model(inp_v)
            v_loss = nn.MSELoss()(preds_v, out_v)
            valid_loss += v_loss.item()
        valid_loss /= len(valid_loader)
        
        writer.add_scalar("valid loss", valid_loss, global_step)
        scheduler.step(valid_loss)
        
        print(f"[Epoch {epoch}] Global Step: {global_step} | "
              f"Train Loss: {epoch_train_loss:.6f} | Valid Loss: {valid_loss:.6f}",
              flush=True)
        
        ### <--- [수정] epoch 50마다만 시각화
        if (epoch + 1) % 50 == 0:
            sample_clean = clean_spectra[0, :]
            sample_noisy = noisy_spectra[0, :]
            plt.figure()
            plt.plot(sample_noisy, label="Noisy Spectrum", color="orange")
            plt.plot(sample_clean, label="Clean Spectrum", color="blue")
            plt.title(f"Sample Prediction at Epoch {epoch+1}")
            plt.legend()
            plt.show()
        
        # Best Models 3개 유지
        if len(best_models) < 3 or valid_loss < max(best_models, key=lambda x: x[0])[0]:
            checkpoint_file = os.path.join(save_model_path, f"{global_step}.pt")
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_step': global_step,
                'loss': loss.item()
            }
            torch.save(state, checkpoint_file)
            best_models.append((valid_loss, epoch, global_step, checkpoint_file))
            print(f"[Save] Checkpoint saved: Global Step {global_step} with Valid Loss {valid_loss:.6f}",
                  flush=True)
            if len(best_models) > 3:
                worst = max(best_models, key=lambda x: x[0])
                best_models.remove(worst)
                if os.path.exists(worst[3]):
                    os.remove(worst[3])
                print(f"[Info] Removed checkpoint {worst[3]} with loss {worst[0]:.6f}", flush=True)
        
        ### <--- [수정] Early Stopping 완전히 제거(학습은 max_epoch까지)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

    # 최종 global_mean과 global_std는 이미 .mat 파일에 업데이트되어 저장됨.
    print("Training finished. Best valid loss =", best_valid_loss)

    
##########################################
# 3. batch_predict 전처리 (학습 시 global 변수 그대로 사용)
##########################################

def merge_txt_to_single_key_mat_1280(base_dir, out_mat, row_points=1600, target_files=1280):
    import os
    import glob
    import numpy as np
    from scipy.interpolate import interp1d
    import scipy.io as sio

    file_list = sorted(glob.glob(os.path.join(base_dir, "**", "*.txt"), recursive=True))
    if not file_list:
        print(f"[오류] '{base_dir}' 내에 .txt 파일이 없습니다.", flush=True)
        return None

    global_min = None
    global_max = None
    data_entries = []
    print("[1] txt 파일 로드 및 전역 x 범위 탐색...", flush=True)
    for fpath in file_list:
        try:
            arr = np.loadtxt(fpath)
        except Exception as e:
            print(f"[로드 오류] {fpath}: {e}", flush=True)
            continue
        if arr.ndim != 2 or arr.shape[1] < 2:
            print(f"[스킵] {fpath}: shape={arr.shape}", flush=True)
            continue
        x = arr[:, 0]
        y = arr[:, 1]
        if x.size < 2:
            print(f"[스킵] {fpath}: 데이터 포인트 부족.", flush=True)
            continue
        lmin, lmax = x.min(), x.max()
        if global_min is None or lmin < global_min:
            global_min = lmin
        if global_max is None or lmax > global_max:
            global_max = lmax
        data_entries.append((fpath, x, y))
    n_files = len(data_entries)
    if n_files == 0:
        print("[결과] 유효한 스펙트럼이 없습니다.", flush=True)
        return None
    print(f"총 {n_files}개의 파일 발견, 전역 x 범위=({global_min}, {global_max})", flush=True)
    x_new = np.linspace(global_min, global_max, row_points)
    data_matrix_all = np.zeros((n_files, row_points), dtype=np.float32)
    print("[2] 각 파일별 cubic 보간 진행 중...", flush=True)
    for i, (fpath, x_arr, y_arr) in enumerate(data_entries):
        try:
            f_intp = interp1d(x_arr, y_arr, kind='cubic', fill_value='extrapolate')
            y_interp = f_intp(x_new)
            data_matrix_all[i, :] = y_interp.astype(np.float32)
        except Exception as e:
            print(f"Interpolation error in {fpath}: {e}", flush=True)
            continue
    if n_files == target_files:
        data_matrix = data_matrix_all
        print(f"파일 수 {n_files}가 target_files와 일치합니다.", flush=True)
    elif n_files > target_files:
        print(f"파일 수 {n_files} > {target_files}. 앞쪽 {target_files}개만 사용.", flush=True)
        data_matrix = data_matrix_all[:target_files, :]
    else:
        diff = target_files - n_files
        print(f"파일 수 {n_files} < {target_files}. 마지막 스펙트럼을 {diff}번 복제합니다.", flush=True)
        data_matrix = np.zeros((target_files, row_points), dtype=np.float32)
        data_matrix[:n_files, :] = data_matrix_all
        last_spec = data_matrix_all[-1, :]
        for i in range(diff):
            data_matrix[n_files + i, :] = last_spec
    print(f"[결과] 최종 data_matrix shape: {data_matrix.shape} (target_files x {row_points})", flush=True)
    mat_dict = {"data_matrix": data_matrix}
    out_dir = os.path.dirname(out_mat)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    sio.savemat(out_mat, mat_dict)
    print(f"[완료] '{out_mat}'에 data_matrix 저장됨.", flush=True)
    return out_mat

def batch_predict(config):
    import torch
    import numpy as np
    import scipy.io as sio
    from scipy.fftpack import dct, idct
    from scipy.linalg import svd

    print("batch predicting...", flush=True)

    merged_mat = os.path.join(config.batch_predict_root, "merged_spectra.mat")
    merge_txt_to_single_key_mat_1280(
        base_dir=config.raw_predict_base,
        out_mat=merged_mat,
        row_points=1600,
        target_files=1280
    )

    if not os.path.exists(merged_mat):
        print(f"[오류] merged_mat 파일 없음: {merged_mat}", flush=True)
        return
    tmp = sio.loadmat(merged_mat, struct_as_record=False, squeeze_me=True)
    if 'data_matrix' not in tmp:
        print("[오류] 'data_matrix' 키가 없음.", flush=True)
        return
    data_matrix = tmp['data_matrix'].astype(np.float64)
    n_files, spec_len = data_matrix.shape
    print(f"[정보] data_matrix shape: ({n_files}, {spec_len})", flush=True)
    if config.visualize_steps:
        plt.figure("Merged Spectra")
        plt.imshow(data_matrix, aspect='auto', cmap='jet')
        plt.title("Merged Spectra")
        plt.colorbar()
        plt.show()

    # 학습 시 저장한 global_mean, global_std를 .mat 파일에서 불러옴
    train_noise_mat = os.path.join(config.train_data_root, "noise_data.mat")
    if not os.path.exists(train_noise_mat):
        print(f"[오류] Global normalization file not found: {train_noise_mat}", flush=True)
        return
    train_mat = sio.loadmat(train_noise_mat, struct_as_record=False, squeeze_me=True)
    if 'global_mean' not in train_mat or 'global_std' not in train_mat:
        print("[오류] Global normalization parameters not found in training file.", flush=True)
        return
    global_mean = float(train_mat['global_mean'])
    global_std = float(train_mat['global_std'])
    print("Loaded global normalization parameters: mean={:.4f}, Global std: {:.4f}".format(global_mean, global_std), flush=True)

    model = eval("{}(1,1)".format(config.model_name))
    model_file = os.path.join(config.test_model_dir, f"{config.global_step}.pt")
    if not os.path.exists(model_file):
        print(f"[오류] 모델 파일 없음: {model_file}", flush=True)
        return
    state = torch.load(model_file, map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    print(f"Successfully loaded model at global step = {state['global_step']}", flush=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("using gpu...", flush=True)

    background_list = []
    baseline_list = []
    noise_pred_list = []
    denoise_list = []
    eps = 1e-6
    lam = 1e5
    p = 0.01
    niter = 10

    for i in range(n_files):
        row_data = data_matrix[i, :]
        baseline = als_baseline(row_data, lam=lam, p=p, niter=niter, visualize=False)
        baseline_list.append(baseline)
        data_bc_removed = row_data - baseline

        overall_progress = ((i+1) / n_files) * 100
        sys.stdout.write(f"\rOverall batch_predict progress: {overall_progress:.1f}% complete")
        sys.stdout.flush()

        U, s, Vt = svd(data_bc_removed.reshape(-1, 1), full_matrices=False)
        background = (U * s) @ Vt
        background = background.flatten()
        background_list.append(background)
        data_bg_removed = data_bc_removed - background

        # DCT 적용 → 그리고 global normalization 적용 (정규화/역정규화 모두 수행)
        resid_dct = dct(data_bg_removed, type=2, norm='ortho')
        norm_val = (resid_dct - global_mean) / (global_std + eps)

        inp_t = norm_val.reshape(1, 1, -1)
        inp_torch = torch.from_numpy(inp_t).float()
        if torch.cuda.is_available():
            inp_torch = inp_torch.cuda()
        with torch.no_grad():
            pred_out = model(inp_torch).cpu().numpy()
        pred_out = pred_out.reshape(-1)
        # 역정규화
        pred_trans = pred_out * (global_std + eps) + global_mean
        noise_pred_1d = idct(pred_trans, type=2, norm='ortho')
        noise_pred_list.append(noise_pred_1d)

        denoised_1d = row_data - noise_pred_1d
        denoise_list.append(denoised_1d)
    print("", flush=True)

    all_pred_noise = np.array(noise_pred_list)
    all_denoised = np.array(denoise_list)
    print("Predicted Noise: mean = {:.4f}, std = {:.4f}, min = {:.4f}, max = {:.4f}".format(
        np.mean(all_pred_noise), np.std(all_pred_noise), np.min(all_pred_noise), np.max(all_pred_noise)), flush=True)
    print("Denoised: mean = {:.4f}, std = {:.4f}, min = {:.4f}, max = {:.4f}".format(
        np.mean(all_denoised), np.std(all_denoised), np.min(all_denoised), np.max(all_denoised)), flush=True)

    tmp['baseline_list'] = np.array(baseline_list, dtype=object)
    tmp['background_list'] = np.array(background_list, dtype=object)
    tmp['orig_spectra'] = data_matrix
    tmp['noise_pred_list'] = np.array(noise_pred_list, dtype=object)
    tmp['denoise_list'] = np.array(denoise_list, dtype=object)

    out_dir = config.batch_save_root
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = os.path.join(out_dir, "batch_predict_result.mat")
    sio.savemat(out_name, tmp, do_compression=True)
    print(f"\n[완료] 최종 결과 저장 -> {out_name}", flush=True)

##########################################
# 5. 체크포인트, 로그, 테스트 결과 저장 경로 함수들
##########################################

def check_dir(config):
    if not os.path.exists(config.checkpoint):
        os.mkdir(config.checkpoint)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    if not os.path.exists(config.test_dir):
        os.mkdir(config.test_dir)

def test_result_dir(config):
    result_dir = os.path.join(config.batch_save_root, config.Instrument,
                              config.model_name, "step_" + str(config.global_step))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir

def save_model_dir(config):
    save_model_dir = os.path.join(config.checkpoint, config.Instrument,
                                  config.model_name, "batch_" + str(config.batch_size))
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    return save_model_dir

def save_log_dir(config):
    save_log_dir = os.path.join(config.logs, config.Instrument,
                                config.model_name, "batch_" + str(config.batch_size))
    if not os.path.exists(save_log_dir):
        os.makedirs(save_log_dir)
    return save_log_dir

##########################################
# 6. main 함수
##########################################

def main(config):
    check_dir(config)
    if config.is_training:
        train(config)
    if config.is_batch_predicting:
        batch_predict(config)

if __name__ == '__main__':
    opt = DefaultConfig()
    opt.visualize_als = False         # ALS 내부 진행률 출력 비활성화 (전체 진행률은 별도 출력)
    opt.visualize_steps = True        # 단계별 시각화 활성화
    # noise_scale는 학습 시 global normalization을 위해 사용됨
    opt.noise_scale = 1.0
    main(opt)
    plt.show()
