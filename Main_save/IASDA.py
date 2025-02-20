import os
import glob
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

# wavelet shrinkage는 이번 코드에서는 사용하지 않습니다.
from Model import UNet_CA, UNet, UNet_CA6, UNet_CA7
from Spectra_generator import spectra_generator
from Make_dataset import Read_data, Make_dataset
from config import DefaultConfig
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# -------------------------------
# ALS Baseline Correction 함수 (희소행렬 기반)
# -------------------------------
def als_baseline(y, lam=1e5, p=0.01, niter=10):
    """
    희소행렬(sparse matrix) 기반 ALS Baseline Correction 함수.
    
    Args:
        y (1D np.array): 입력 신호.
        lam (float): baseline의 부드러움을 제어 (큰 값일수록 baseline이 부드러워짐).
        p (float): 비대칭성 파라미터 (보통 0.01~0.05).
        niter (int): 반복 횟수.
        
    Returns:
        baseline (1D np.array): 추정된 baseline.
    """
    L = len(y)
    # 2차 미분을 위한 희소 행렬 D 생성 (각 행: [1, -2, 1])
    diagonals = [np.ones(L), -2 * np.ones(L), np.ones(L)]
    offsets = [0, 1, 2]
    D = diags(diagonals, offsets, shape=(L - 2, L))
    # D.T @ D는 (L x L) 크기의 희소 행렬 (2차 미분 제곱항)
    DTD = D.T @ D
    w = np.ones(L)
    for i in range(niter):
        W = diags(w, 0)
        Z = spsolve(W + lam * DTD, w * y)
        w = p * (y > Z) + (1 - p) * (y < Z)
    return Z

# -------------------------------
# arcsinh 변환 + Z-score 정규화 함수
# -------------------------------
def arcsinh_transform(signal, eps=1e-6):
    """
    입력 신호에 arcsinh 변환을 적용합니다.
    arcsinh는 음수에도 자연스럽게 정의되므로 별도의 부호 보존 처리가 필요 없습니다.
    """
    return np.arcsinh(signal)

def arcsinh_zscore_normalize(signal, eps=1e-6):
    """
    입력 신호에 대해 arcsinh 변환을 적용한 후, 평균과 표준편차를 이용하여 Z-score 정규화합니다.
    
    Returns:
        norm_signal: 정규화된 신호
        mean_val: arcsinh 변환 후의 평균
        std_val: arcsinh 변환 후의 표준편차
    """
    signal_trans = arcsinh_transform(signal, eps=eps)
    mean_val = np.mean(signal_trans)
    std_val = np.std(signal_trans)
    norm_signal = (signal_trans - mean_val) / (std_val + eps)
    return norm_signal, mean_val, std_val

# -------------------------------
# pipeline_and_save_noise 함수
# (Interpolation → ALS Baseline Correction → SVD를 통한 BG 제거 → DCT → Transpose →
#  arcsinh 변환 + Z-score 정규화 → .mat 저장)
# -------------------------------
def pipeline_and_save_noise(base_folder, output_mat, config, subfolders=("1", "2", "3"),
                              zero_cut=0.0, row_new_length=1600, col_new_length=640, remove_svs=1):
    """
    1) base_folder 내의 txt 파일을 로드하여 row 방향 cubic 보간 수행 → all_spectra (shape: row_new_length x num_files)
    2) 열 방향 보간으로 2D 행렬 생성 → all_spectra_2d (shape: row_new_length x col_new_length)
    3) 각 열에 대해 ALS baseline correction 적용 → data_bc_removed
    4) baseline 제거된 데이터에 대해 SVD를 적용하여 배경 제거 → data_bg_removed
    5) 각 열에 대해 DCT 적용 → dct_result
    6) 행렬 전치하여 noise_data (shape: col_new_length x row_new_length) 생성
    7) 각 행에 대해 arcsinh 변환 + Z-score 정규화 적용 → norm_data (최종 noise 데이터)
    8) 결과를 .mat 파일에 저장 (키: 'noise', 'train_means', 'train_stds')
    """
    # --- Step 1: Row Interpolation ---
    txt_files = []
    for sf in subfolders:
        sf_path = os.path.join(base_folder, sf)
        fpaths = glob.glob(os.path.join(sf_path, "*.txt"))
        txt_files.extend(fpaths)
    if not txt_files:
        print("No txt files found in subfolders:", subfolders)
        return None
    txt_files = sorted(txt_files)
    spectra_list = []
    for fpath in txt_files:
        try:
            arr = np.loadtxt(fpath)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            continue
        if arr.ndim != 2 or arr.shape[1] < 2:
            print(f"Skipping {fpath}, shape={arr.shape}")
            continue
        x_raw, y_raw = arr[:, 0], arr[:, 1]
        mask = (x_raw >= zero_cut)
        x_cut = x_raw[mask]
        y_cut = y_raw[mask]
        if x_cut.size < 2:
            print(f"Skipping {fpath}: not enough data")
            continue
        x_min, x_max = x_cut.min(), x_cut.max()
        if x_min == x_max:
            continue
        x_new = np.linspace(x_min, x_max, row_new_length)
        try:
            f_intp = interp1d(x_cut, y_cut, kind='cubic')
            y_new = f_intp(x_new)
        except Exception as e:
            print(f"Interpolation error in {fpath}: {e}")
            continue
        spectra_list.append(y_new)
    if not spectra_list:
        print("No valid spectra found.")
        return None
    all_spectra = np.array(spectra_list).T  # shape: (row_new_length, num_files)
    print("After row interpolation:", all_spectra.shape)
    plt.figure()
    for idx in range(min(3, all_spectra.shape[1])):
        plt.plot(all_spectra[:, idx], label=f"Sample {idx+1}")
    plt.title("After Row Interpolation")
    plt.xlabel("Row Index")
    plt.ylabel("Value")
    plt.legend()
    # plt.show()
    
    # --- Step 2: Column Interpolation ---
    num_rows, num_cols = all_spectra.shape
    col_idx = np.arange(num_cols)
    col_new = np.linspace(0, num_cols - 1, col_new_length)
    all_spectra_2d = np.zeros((num_rows, col_new_length), dtype=all_spectra.dtype)
    for i in range(num_rows):
        f_c = interp1d(col_idx, all_spectra[i, :], kind='linear')
        all_spectra_2d[i, :] = f_c(col_new)
    print("After column interpolation:", all_spectra_2d.shape)
    plt.figure()
    for idx in range(min(3, all_spectra_2d.shape[1])):
        plt.plot(all_spectra_2d[:, idx], label=f"Sample {idx+1}")
    plt.title("After Column Interpolation")
    plt.xlabel("Row Index")
    plt.ylabel("Value")
    plt.legend()
    # plt.show()
    
    # --- Step 3: ALS Baseline Correction ---
    data_baseline = np.zeros_like(all_spectra_2d)
    for j in range(all_spectra_2d.shape[1]):
        baseline = als_baseline(all_spectra_2d[:, j], lam=1e5, p=0.01, niter=10)
        data_baseline[:, j] = baseline
    data_bc_removed = all_spectra_2d - data_baseline
    print("After ALS baseline correction:", data_bc_removed.shape)
    plt.figure()
    for idx in range(min(3, data_baseline.shape[1])):
        plt.plot(data_baseline[:, idx], label=f"Baseline Sample {idx+1}")
    plt.title("Estimated Baseline (ALS)")
    plt.xlabel("Row Index")
    plt.ylabel("Baseline Value")
    plt.legend()
    # plt.show()
    plt.figure()
    for idx in range(min(3, data_bc_removed.shape[1])):
        plt.plot(data_bc_removed[:, idx], label=f"Corrected Sample {idx+1}")
    plt.title("After Baseline Removal (ALS)")
    plt.xlabel("Row Index")
    plt.ylabel("Corrected Value")
    plt.legend()
    # plt.show()
    
    # --- Step 4: SVD Background Removal ---
    U, s, Vt = svd(data_bc_removed, full_matrices=False)
    s_mod = s.copy()
    for i in range(remove_svs):
        if i < len(s_mod):
            s_mod[i] *= config.fade_factor
    data_bg_removed = U @ np.diag(s_mod) @ Vt
    print("After SVD background removal:", data_bg_removed.shape)
    plt.figure()
    for idx in range(min(3, data_bg_removed.shape[1])):
        plt.plot(data_bg_removed[:, idx], label=f"Sample {idx+1}")
    plt.title("After SVD Background Removal")
    plt.xlabel("Row Index")
    plt.ylabel("Value")
    plt.legend()
    # plt.show()
    
    # --- Step 5: DCT 적용 ---
    dct_result = np.zeros_like(data_bg_removed)
    for j in range(data_bg_removed.shape[1]):
        dct_result[:, j] = dct(data_bg_removed[:, j], type=2, norm='ortho')
    print("After DCT:", dct_result.shape)
    plt.figure()
    for idx in range(min(3, dct_result.shape[1])):
        plt.plot(dct_result[:, idx], label=f"Sample {idx+1}")
    plt.title("After DCT")
    plt.xlabel("Frequency Index")
    plt.ylabel("DCT Coefficient")
    plt.legend()
    # plt.show()
    
    # --- Step 6: Transpose ---
    noise_data = dct_result.T
    print("After transpose:", noise_data.shape)
    plt.figure()
    for idx in range(min(3, noise_data.shape[0])):
        plt.plot(noise_data[idx, :], label=f"Sample {idx+1}")
    plt.title("After Transpose")
    plt.xlabel("Spectral Point")
    plt.ylabel("Value")
    plt.legend()
    # plt.show()
    
    # --- Step 7: arcsinh 변환 + Z-score 정규화 ---
    eps = 1e-6
    norm_data = np.zeros_like(noise_data)
    norm_means = []
    norm_stds = []
    for i in range(noise_data.shape[0]):
        norm_row, mean_val, std_val = arcsinh_zscore_normalize(noise_data[i, :], eps=eps)
        norm_data[i, :] = norm_row
        norm_means.append(mean_val)
        norm_stds.append(std_val)
    print("After arcsinh-Zscore Normalization:", norm_data.shape)
    plt.figure()
    for idx in range(min(3, norm_data.shape[0])):
        plt.plot(norm_data[idx, :], label=f"Sample {idx+1}")
    plt.title("After arcsinh-Zscore Normalization")
    plt.xlabel("Spectral Point")
    plt.ylabel("Normalized Value")
    plt.legend()
    # plt.show()
    
    # --- Step 8: 결과 저장 ---
    out_dir = os.path.dirname(output_mat)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    savemat(output_mat, {"noise": norm_data,
                         "train_means": np.array(norm_means),
                         "train_stds": np.array(norm_stds)})
    print("Saved noise_data.mat =>", output_mat)
    return output_mat

# ============================================================================
# [Train 단계]
# Noise에 GT 생성 및 추가 후 U-Net 모델 학습
# ============================================================================
def train(config):
    """
    1) pipeline_and_save_noise()를 호출하여 noise_data.mat를 생성  
    2) 생성된 .mat 파일을 이용하여 Read_data, Make_dataset로 데이터셋 구성  
    3) U-Net 모델 학습 진행
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # (A) 노이즈 데이터 전처리 및 .mat 저장
    output_mat = os.path.join(config.train_data_root, "noise_data.mat")
    print("[INFO] Start pipeline_and_save_noise ...")
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
    print("[INFO] Noise data .mat 생성 완료 =>", output_mat)
    
    # (B) U-Net 모델 생성
    model = eval("{}(1,1)".format(config.model_name))
    print(model)
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
        model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    schedual = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0005)
    
    save_model_path = save_model_dir(config)
    save_log = save_log_dir(config)
    global_step = 0
    
    # 사전 학습된 모델 로드 (옵션)
    if config.is_pretrain:
        global_step = config.global_step
        model_file = os.path.join(save_model_path, f"{global_step}.pt")
        kwargs = {'map_location': lambda storage, loc: storage.cuda([0, 1])}
        state = torch.load(model_file, **kwargs)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print('Loaded pretrained model at global step =', global_step)
    
    # (C) DataLoader 설정 (noise_data.mat 이용)
    reader = Read_data(config.train_data_root, config.valid_ratio)
    train_set, valid_set = reader.read_file()
    TrainSet, ValidSet = Make_dataset(train_set), Make_dataset(valid_set)
    train_loader = DataLoader(TrainSet, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(ValidSet, batch_size=256, pin_memory=True)
    
    gen_train = spectra_generator()
    gen_valid = spectra_generator()
    writer = SummaryWriter(save_log)
    
    # (D) Training Loop
    for epoch in range(config.max_epoch):
        model.train()
        for idx, noise in enumerate(train_loader):
            noise = noise.squeeze().numpy()  # (batch_size, spec)
            spectra_num, spec = noise.shape
    
            clean_spectra = gen_train.generator(spec, spectra_num).T
            noisy_spectra = clean_spectra + noise
    
            input_coef = np.zeros_like(noisy_spectra)
            output_coef = np.zeros_like(noise)
            for i in range(spectra_num):
                input_coef[i, :] = dct(noisy_spectra[i, :], norm='ortho')
                output_coef[i, :] = dct(noise[i, :], norm='ortho')
    
            input_coef = input_coef.reshape(-1, 1, spec).astype(np.float32)
            output_coef = output_coef.reshape(-1, 1, spec).astype(np.float32)
    
            inp_t = torch.from_numpy(input_coef)
            out_t = torch.from_numpy(output_coef)
            if torch.cuda.is_available():
                inp_t, out_t = inp_t.cuda(), out_t.cuda()
    
            global_step += 1
            preds = model(inp_t)
            loss = criterion(preds, out_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if idx % config.print_freq == 0:
                print(f"[Train] epoch={epoch}, batch={idx}, global_step={global_step}, loss={loss.item()}")
            writer.add_scalar("train loss", loss.item(), global_step)
    
        # Validation
        model.eval()
        valid_loss = 0.0
        for idx_v, noise_v in enumerate(valid_loader):
            noise_v = noise_v.squeeze().numpy()  # (batch_size, spec)
            spectra_num, spec = noise_v.shape
    
            clean_spectra = gen_valid.generator(spec, spectra_num).T
            noisy_spectra = clean_spectra + noise_v
    
            input_coef_v = np.zeros_like(noisy_spectra)
            output_coef_v = np.zeros_like(noise_v)
            for i in range(spectra_num):
                input_coef_v[i, :] = dct(noisy_spectra[i, :], norm='ortho')
                output_coef_v[i, :] = dct(noise_v[i, :], norm='ortho')
    
            inp_v = torch.from_numpy(input_coef_v.reshape(-1, 1, spec).astype(np.float32))
            out_v = torch.from_numpy(output_coef_v.reshape(-1, 1, spec).astype(np.float32))
            if torch.cuda.is_available():
                inp_v, out_v = inp_v.cuda(), out_v.cuda()
    
            with torch.no_grad():
                preds_v = model(inp_v)
                v_loss = criterion(preds_v, out_v)
            valid_loss += v_loss.item()
    
        valid_loss /= len(valid_loader)
        writer.add_scalar("valid loss", valid_loss, global_step)
        schedual.step()
    
        if (epoch+1) % 100 == 0:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_step': global_step,
                'loss': loss.item()
            }
            model_file = os.path.join(save_model_path, f"{global_step}.pt")
            torch.save(state, model_file)
            print(f"[Save] epoch={epoch+1}, model_file={model_file}")

# ============================================================================
# [Batch Predict 전처리]
# (merge_txt_to_single_key_mat_1280 : (1276, 1039) → (1280, 1600)로 shape 변환)
# ============================================================================
def merge_txt_to_single_key_mat_1280(base_dir, out_mat, row_points=1600, target_files=1280):
    """
    base_dir 내의 모든 txt 파일을 찾아, 각 파일의 (x,y) 데이터를 로드한 후,
    전역 x 범위에 대해 row_points 개의 점으로 cubic 보간하여 (n_files, row_points) 배열을 생성.
    파일 수가 target_files보다 적으면 마지막 스펙트럼을 복제, 많으면 앞쪽 target_files 개만 사용하여
    최종 (1280, row_points) 크기의 data_matrix를 .mat 파일로 저장.
    """
    file_list = sorted(glob.glob(os.path.join(base_dir, "**", "*.txt"), recursive=True))
    if not file_list:
        print(f"[오류] '{base_dir}' 내에 .txt 파일이 없습니다.")
        return None

    global_min = None
    global_max = None
    data_entries = []
    print("[1] txt 파일 로드 및 전역 x 범위 탐색...")
    for fpath in file_list:
        try:
            arr = np.loadtxt(fpath)
        except Exception as e:
            print(f"[로드 오류] {fpath}: {e}")
            continue
        if arr.ndim != 2 or arr.shape[1] < 2:
            print(f"[스킵] {fpath}: shape={arr.shape}")
            continue
        x = arr[:, 0]
        y = arr[:, 1]
        if x.size < 2:
            print(f"[스킵] {fpath}: 데이터 포인트 부족.")
            continue
        lmin, lmax = x.min(), x.max()
        if global_min is None or lmin < global_min:
            global_min = lmin
        if global_max is None or lmax > global_max:
            global_max = lmax
        data_entries.append((fpath, x, y))
    n_files = len(data_entries)
    if n_files == 0:
        print("[결과] 유효한 스펙트럼이 없습니다.")
        return None
    print(f"총 {n_files}개의 파일 발견, 전역 x 범위=({global_min}, {global_max})")
    x_new = np.linspace(global_min, global_max, row_points)
    data_matrix_all = np.zeros((n_files, row_points), dtype=np.float32)
    print("[2] 각 파일별 cubic 보간 진행 중...")
    for i, (fpath, x_arr, y_arr) in enumerate(data_entries):
        f_intp = interp1d(x_arr, y_arr, kind='cubic', fill_value='extrapolate')
        y_interp = f_intp(x_new)
        data_matrix_all[i, :] = y_interp.astype(np.float32)
    if n_files == target_files:
        data_matrix = data_matrix_all
        print(f"파일 수 {n_files}가 target_files와 일치합니다.")
    elif n_files > target_files:
        print(f"파일 수 {n_files} > {target_files}. 앞쪽 {target_files}개만 사용.")
        data_matrix = data_matrix_all[:target_files, :]
    else:
        diff = target_files - n_files
        print(f"파일 수 {n_files} < {target_files}. 마지막 스펙트럼을 {diff}번 복제합니다.")
        data_matrix = np.zeros((target_files, row_points), dtype=np.float32)
        data_matrix[:n_files, :] = data_matrix_all
        last_spec = data_matrix_all[-1, :]
        for i in range(diff):
            data_matrix[n_files + i, :] = last_spec
    print(f"[결과] 최종 data_matrix shape: {data_matrix.shape} (1280 x {row_points})")
    mat_dict = {"data_matrix": data_matrix}
    out_dir = os.path.dirname(out_mat)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    sio.savemat(out_mat, mat_dict)
    print(f"[완료] '{out_mat}'에 data_matrix 저장됨.")
    return out_mat

# ============================================================================
# [Batch Predict 단계]
# (ALS baseline correction → SVD BG 제거 → DCT → arcsinh+Zscore 정규화 →
#  모델 예측 → Inverse Normalization (역 arcsinh 적용) → IDCT →
#  SVD BG 및 ALS baseline 재결합 → predicted noise, denoised spectrum 저장)
# ============================================================================
def batch_predict(config):
    """
    1) base_folder 내의 모든 txt 파일들을 통합하여 .mat 파일 생성  
       (merge_txt_to_single_key_mat_1280 이용 → shape: (1280,1600))
    2) 저장된 .mat 파일 로드 (키: 'data_matrix')
    3) 각 스펙트럼(행)에 대해:
         (a) ALS Baseline Correction 적용, 제거한 baseline 저장
         (b) SVD를 통한 BG 제거 및 제거한 BG 저장
         (c) 행 단위 DCT 적용
         (d) arcsinh 변환 + Zscore 정규화 적용
         (e) 모델 예측 → 예측된 노이즈 DCT 계수 획득
         (f) Inverse Normalization: (sample_std, sample_mean)를 곱/더한 후, np.sinh를 적용하여 역 arcsinh 변환
         (g) IDCT를 통해 노이즈 복원
         (h) 원본 스펙트럼에서 예측된 노이즈 제거 후, ALS baseline과 SVD BG를 재결합하여 denoised 스펙트럼 생성
    4) 결과를 .mat 파일로 저장
    """
    import torch
    import numpy as np
    import scipy.io as sio
    from scipy.fftpack import dct, idct
    from scipy.linalg import svd

    print("batch predicting...")

    # (1) 텍스트 파일 병합 → .mat 생성
    merged_mat = os.path.join(config.batch_predict_root, "merged_spectra.mat")
    merge_txt_to_single_key_mat_1280(
        base_dir=config.raw_predict_base,
        out_mat=merged_mat,
        row_points=1600,
        target_files=1280
    )

    # (2) 저장된 .mat 파일 로드
    if not os.path.exists(merged_mat):
        print(f"[오류] merged_mat 파일 없음: {merged_mat}")
        return
    tmp = sio.loadmat(merged_mat, struct_as_record=False, squeeze_me=True)
    if 'data_matrix' not in tmp:
        print("[오류] 'data_matrix' 키가 없음.")
        return
    data_matrix = tmp['data_matrix'].astype(np.float64)
    n_files, spec_len = data_matrix.shape
    print(f"[정보] data_matrix shape: ({n_files},{spec_len})")

    # (3) 모델 로드
    model = eval("{}(1,1)".format(config.model_name))
    model_file = os.path.join(config.test_model_dir, f"{config.global_step}.pt")
    if not os.path.exists(model_file):
        print(f"[오류] 모델 파일 없음: {model_file}")
        return
    state = torch.load(model_file, map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    print(f"Successfully loaded model at global step = {state['global_step']}")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("using gpu...")

    # (4) 각 스펙트럼 처리
    background_list = []   # SVD에서 제거한 BG 저장
    baseline_list = []     # ALS baseline에서 제거한 baseline 저장
    noise_pred_list = []
    denoise_list = []
    eps = 1e-6
    lam = 1e5
    p = 0.01
    niter = 10

    for i in range(n_files):
        row_data = data_matrix[i, :]
        # (a) ALS Baseline Correction
        baseline = als_baseline(row_data, lam=lam, p=p, niter=niter)
        baseline_list.append(baseline)
        data_bc_removed = row_data - baseline

        # (b) SVD Background 제거
        U, s, Vt = svd(data_bc_removed.reshape(-1, 1), full_matrices=False)
        background = (U * s) @ Vt
        background = background.flatten()
        background_list.append(background)
        data_bg_removed = data_bc_removed - background

        # (c) 행 단위 DCT 적용
        resid_dct = dct(data_bg_removed, type=2, norm='ortho')
        
        # (d) arcsinh 변환 + Zscore 정규화 적용
        norm_val, sample_mean, sample_std = arcsinh_zscore_normalize(resid_dct, eps=eps)
        
        # (e) 모델 예측 (입력 shape: (1,1,spec))
        inp_t = norm_val.reshape(1, 1, -1)
        inp_torch = torch.from_numpy(inp_t).float()
        if torch.cuda.is_available():
            inp_torch = inp_torch.cuda()
        with torch.no_grad():
            pred_out = model(inp_torch).cpu().numpy()
        pred_out = pred_out.reshape(-1)
        
        # (f) Inverse Normalization: (sample_std, sample_mean) 역정규화 후, np.sinh로 역 arcsinh 변환
        pred_trans = pred_out * (sample_std + eps) + sample_mean
        pred_dct = np.sinh(pred_trans)
        
        # (g) IDCT를 통해 노이즈 복원
        noise_pred_1d = idct(pred_dct, type=2, norm='ortho')
        noise_pred_list.append(noise_pred_1d)
        
        # (h) 최종 denoised 스펙트럼 생성: 원본 스펙트럼에서 예측된 노이즈 제거 후, baseline과 BG 재결합
        denoised_1d = row_data - noise_pred_1d + baseline + background
        denoise_list.append(denoised_1d)

    # (5) 결과 저장
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
    print(f"[완료] 최종 결과 저장 -> {out_name}")

# ============================================================================
# 체크포인트, 로그, 테스트 결과 저장 경로 함수들
# ============================================================================
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

# ============================================================================
# main 함수
# ============================================================================
def main(config):
    check_dir(config)
    if config.is_training:
        train(config)
    if config.is_batch_predicting:
        batch_predict(config)

if __name__ == '__main__':
    opt = DefaultConfig()
    main(opt)
    plt.show()
