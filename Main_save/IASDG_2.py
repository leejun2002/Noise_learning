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

from multiprocessing import Pool

# -------------------------------
# ALS Baseline Correction 함수 (np.linalg.solve() 사용)
# -------------------------------
def als_baseline(y, lam=1e5, p=0.01, niter=10):
    """
    Asymmetric Least Squares (ALS) baseline correction using dense matrix computation.
    
    Args:
        y (1D np.array): 입력 신호.
        lam (float): baseline의 부드러움을 제어 (큰 값일수록 baseline이 부드러워짐).
        p (float): 비대칭성 파라미터 (보통 0.01~0.05).
        niter (int): 반복 횟수.
        
    Returns:
        baseline (1D np.array): 추정된 baseline.
    """
    L = len(y)
    # 2차 미분 행렬 계산: 단위행렬에서 2차 차분 후, lam을 곱함.
    D = np.diff(np.eye(L), 2)
    D = lam * D.dot(D.T)
    w = np.ones(L)
    for i in range(niter):
        W = np.diag(w)
        # (W + D)*Z = w*y 를 np.linalg.solve로 풀어 baseline(Z)를 구함.
        Z = np.linalg.solve(W + D, w * y)
        w = p * (y > Z) + (1 - p) * (y < Z)
    return Z

# -------------------------------
# Instance Normalization 함수 (각 스펙트럼별)
# -------------------------------
def instance_normalize(signal, eps=1e-6):
    """
    입력 신호(1D 배열)를 해당 신호의 평균과 표준편차를 이용하여 정규화합니다.
    
    Returns:
        norm_signal: 정규화된 신호
        mean_val: 입력 신호의 평균
        std_val: 입력 신호의 표준편차
    """
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    norm_signal = (signal - mean_val) / (std_val + eps)
    return norm_signal, mean_val, std_val

# -------------------------------
# 개별 텍스트 파일 처리 함수 (병렬 처리용)
# -------------------------------
def process_file(fpath, zero_cut, row_new_length):
    """
    주어진 텍스트 파일을 로드하여, x와 y 데이터를 추출한 후,
    x가 zero_cut 이상인 부분을 대상으로 cubic 보간을 수행하여
    row_new_length 길이의 y_new 벡터를 반환합니다.
    실패 시 None 반환.
    """
    try:
        arr = np.loadtxt(fpath)
    except Exception as e:
        print(f"Error loading {fpath}: {e}")
        return None
    if arr.ndim != 2 or arr.shape[1] < 2:
        print(f"Skipping {fpath}, shape={arr.shape}")
        return None
    x_raw, y_raw = arr[:, 0], arr[:, 1]
    mask = (x_raw >= zero_cut)
    x_cut = x_raw[mask]
    y_cut = y_raw[mask]
    if x_cut.size < 2:
        print(f"Skipping {fpath}: not enough data")
        return None
    x_min, x_max = x_cut.min(), x_cut.max()
    if x_min == x_max:
        return None
    x_new = np.linspace(x_min, x_max, row_new_length)
    try:
        f_intp = interp1d(x_cut, y_cut, kind='cubic')
        y_new = f_intp(x_new)
    except Exception as e:
        print(f"Interpolation error in {fpath}: {e}")
        return None
    return y_new

# -------------------------------
# pipeline_and_save_noise 함수 (병렬 처리 + 전처리 및 시각화)
# -------------------------------
def pipeline_and_save_noise(base_folder, output_mat, config, subfolders=("1", "2", "3"),
                              zero_cut=0.0, row_new_length=1600, col_new_length=640, remove_svs=1):
    """
    1) base_folder 내의 txt 파일들을 읽어 행 보간(병렬 처리) 수행 → all_spectra, all_spectra_2d  
    2) ALS baseline correction 적용 → data_bc_removed  
    3) SVD로 BG 제거 → data_bg_removed  
    4) 각 열에 대해 DCT 적용 → dct_result  
    5) 행렬 전치하여 noise_data 생성  
    6) 전역 정규화: 전체 noise_data에 대해 평균, std를 구해 (noise - mean)/std 수행  
    7) norm_data와 전역 정규화 파라미터(global_mean, global_std)를 .mat 파일에 저장  
    8) 각 단계 결과를 하나의 Figure에 Subplot 시각화  
    """
    eps = 1e-6

    # --- Step 1: Row Interpolation (병렬 처리) ---
    txt_files = []
    for sf in subfolders:
        sf_path = os.path.join(base_folder, sf)
        fpaths = glob.glob(os.path.join(sf_path, "*.txt"))
        txt_files.extend(fpaths)
    if not txt_files:
        print("No txt files found in subfolders:", subfolders)
        return None
    txt_files = sorted(txt_files)
    
    with Pool() as pool:
        results = pool.starmap(process_file, [(fpath, zero_cut, row_new_length) for fpath in txt_files])
    spectra_list = [res for res in results if res is not None]
    
    if not spectra_list:
        print("No valid spectra found.")
        return None
    all_spectra = np.array(spectra_list).T  # shape: [row_new_length, num_files]
    print("After row interpolation:", all_spectra.shape)

    # --- Step 2: Column Interpolation ---
    num_rows, num_cols = all_spectra.shape
    col_idx = np.arange(num_cols)
    col_new = np.linspace(0, num_cols - 1, col_new_length)
    all_spectra_2d = np.zeros((num_rows, col_new_length), dtype=all_spectra.dtype)
    for i in range(num_rows):
        f_c = interp1d(col_idx, all_spectra[i, :], kind='linear')
        all_spectra_2d[i, :] = f_c(col_new)
    print("After column interpolation:", all_spectra_2d.shape)

    # --- Step 3: ALS Baseline Correction ---
    data_baseline = np.zeros_like(all_spectra_2d)
    for j in range(all_spectra_2d.shape[1]):
        baseline = als_baseline(all_spectra_2d[:, j], lam=1e5, p=0.01, niter=10)
        data_baseline[:, j] = baseline
    data_bc_removed = all_spectra_2d - data_baseline
    print("After ALS baseline correction:", data_bc_removed.shape)

    # --- Step 4: SVD Background Removal ---
    U, s, Vt = svd(data_bc_removed, full_matrices=False)
    s_mod = s.copy()
    for i in range(remove_svs):
        if i < len(s_mod):
            s_mod[i] *= config.fade_factor
    data_bg_removed = U @ np.diag(s_mod) @ Vt
    print("After SVD background removal:", data_bg_removed.shape)

    # --- Step 5: DCT 적용 ---
    dct_result = np.zeros_like(data_bg_removed)
    for j in range(data_bg_removed.shape[1]):
        dct_result[:, j] = dct(data_bg_removed[:, j], type=2, norm='ortho')
    print("After DCT:", dct_result.shape)

    # --- Step 6: Transpose ---
    noise_data = dct_result.T  # shape: [col_new_length, row_new_length]
    print("After transpose:", noise_data.shape)

    # --- Step 7: Global Normalization ---
    global_mean = np.mean(noise_data)
    global_std = np.std(noise_data)
    norm_data = (noise_data - global_mean) / (global_std + eps)
    print("After global normalization:", norm_data.shape)
    print("Global mean: {:.4f}, Global std: {:.4f}".format(global_mean, global_std))
    
    # --- Subplot 시각화 ---
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    axs = axs.flatten()

    x_axis = np.arange(all_spectra.shape[0])
    axs[0].plot(x_axis, all_spectra[:, 0], label="Sample 1")
    axs[0].plot(x_axis, all_spectra[:, 1], label="Sample 2")
    axs[0].plot(x_axis, all_spectra[:, 2], label="Sample 3")
    axs[0].set_title("After Row Interpolation")
    axs[0].legend()

    axs[1].plot(x_axis, all_spectra_2d[:, 0], label="Sample 1")
    axs[1].plot(x_axis, all_spectra_2d[:, 1], label="Sample 2")
    axs[1].plot(x_axis, all_spectra_2d[:, 2], label="Sample 3")
    axs[1].set_title("After Column Interpolation")
    axs[1].legend()

    axs[2].plot(x_axis, data_baseline[:, 0], label="Sample 1")
    axs[2].plot(x_axis, data_baseline[:, 1], label="Sample 2")
    axs[2].plot(x_axis, data_baseline[:, 2], label="Sample 3")
    axs[2].set_title("Estimated Baseline (ALS)")
    axs[2].legend()

    axs[3].plot(x_axis, data_bc_removed[:, 0], label="Sample 1")
    axs[3].plot(x_axis, data_bc_removed[:, 1], label="Sample 2")
    axs[3].plot(x_axis, data_bc_removed[:, 2], label="Sample 3")
    axs[3].set_title("After Baseline Removal (ALS)")
    axs[3].legend()

    axs[4].plot(x_axis, data_bg_removed[:, 0], label="Sample 1")
    axs[4].plot(x_axis, data_bg_removed[:, 1], label="Sample 2")
    axs[4].plot(x_axis, data_bg_removed[:, 2], label="Sample 3")
    axs[4].set_title("After SVD BG Removal")
    axs[4].legend()

    freq_axis = np.arange(dct_result.shape[0])
    axs[5].plot(freq_axis, dct_result[:, 0], label="Sample 1")
    axs[5].plot(freq_axis, dct_result[:, 1], label="Sample 2")
    axs[5].plot(freq_axis, dct_result[:, 2], label="Sample 3")
    axs[5].set_title("After DCT")
    axs[5].legend()

    spectral_axis = np.arange(noise_data.shape[1])
    axs[6].plot(spectral_axis, noise_data[0, :], label="Sample 1")
    axs[6].plot(spectral_axis, noise_data[1, :], label="Sample 2")
    axs[6].plot(spectral_axis, noise_data[2, :], label="Sample 3")
    axs[6].set_title("After Transpose")
    axs[6].legend()

    axs[7].plot(spectral_axis, norm_data[0, :], label="Sample 1")
    axs[7].plot(spectral_axis, norm_data[1, :], label="Sample 2")
    axs[7].plot(spectral_axis, norm_data[2, :], label="Sample 3")
    axs[7].set_title("After Global Normalization")
    axs[7].legend()

    axs[8].axis('off')
    plt.tight_layout()
    plt.show()

    # --- Step 8: 결과 저장 ---
    out_dir = os.path.dirname(output_mat)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    savemat(output_mat, {"noise": norm_data,
                         "global_mean": global_mean,
                         "global_std": global_std})
    print("Saved noise_data.mat =>", output_mat)
    return output_mat

plt.show()


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
    plt.show()  # 전처리 플롯들을 한 창에 표시
    print("[INFO] Noise data .mat 생성 완료 =>", output_mat)
    
    # (B) U-Net 모델 생성
    model = eval("{}(1,1)".format(config.model_name))
    print(model)
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
        model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0005)
    
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
        scheduler.step()
    
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
# [Batch Predict 전처리 단계]
# (merge_txt_to_single_key_mat_1280: (1276, 1039) → (1280, 1600)로 shape 변환)
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
# (ALS baseline correction → SVD BG 제거 → DCT → Instance normalization →
#  모델 예측 → 역 정규화 → IDCT → SVD BG 및 ALS baseline 재결합 →
#  predicted noise, denoised spectrum 저장)
# ============================================================================

def batch_predict(config):
    """
    1) base_folder 내의 모든 txt 파일들을 병합하여 merged_spectra.mat 생성  
       (merge_txt_to_single_key_mat_1280 이용)
    2) 저장된 merged_spectra.mat (키: 'data_matrix')를 로드  
    3) 각 스펙트럼에 대해:
         (a) ALS Baseline Correction → baseline 제거 및 저장  
         (b) SVD BG 제거 → BG 제거 및 저장  
         (c) 행 단위 DCT 적용 → resid_dct 획득  
         (d) 전역 정규화: resid_dct를 저장된 global_mean, global_std로 정규화  
         (e) 모델 예측 → 예측된 노이즈 DCT 계수 획득  
         (f) 역 정규화: 예측값에 global_std, global_mean 적용 후 np.sinh 제거 (선형 변환 사용)  
         (g) IDCT 적용하여 노이즈 복원  
         (h) 원본 스펙트럼에서 예측 노이즈 제거 후 baseline과 BG 재결합하여 denoised spectrum 생성  
    4) 결과를 .mat 파일에 저장
    """
    import torch
    import numpy as np
    import scipy.io as sio
    from scipy.fftpack import dct, idct
    from scipy.linalg import svd

    print("batch predicting...")

    # (1) 텍스트 파일 병합 → merged_spectra.mat 생성
    merged_mat = os.path.join(config.batch_predict_root, "merged_spectra.mat")
    merge_txt_to_single_key_mat_1280(
        base_dir=config.raw_predict_base,
        out_mat=merged_mat,
        row_points=1600,
        target_files=1280
    )

    # (2) 저장된 merged_spectra.mat 로드
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

    # (2.5) 학습 시 저장한 전역 정규화 파라미터(global_mean, global_std) 불러오기
    train_noise_mat = os.path.join(config.train_data_root, "noise_data.mat")
    if not os.path.exists(train_noise_mat):
        print(f"[오류] Global normalization file not found: {train_noise_mat}")
        return
    train_mat = sio.loadmat(train_noise_mat, struct_as_record=False, squeeze_me=True)
    if 'global_mean' not in train_mat or 'global_std' not in train_mat:
        print("[오류] Global normalization parameters not found in training file.")
        return
    global_mean = float(train_mat['global_mean'])
    global_std = float(train_mat['global_std'])
    print("Loaded global normalization parameters: mean={:.4f}, std={:.4f}".format(global_mean, global_std))

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
    background_list = []   # SVD로 제거한 BG 저장
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
        
        # (d) 전역 정규화: 학습 시 구한 global_mean, global_std 사용
        norm_val = (resid_dct - global_mean) / (global_std + eps)
        
        # (e) 모델 예측 (입력 shape: (1,1,spec))
        inp_t = norm_val.reshape(1, 1, -1)
        inp_torch = torch.from_numpy(inp_t).float()
        if torch.cuda.is_available():
            inp_torch = inp_torch.cuda()
        with torch.no_grad():
            pred_out = model(inp_torch).cpu().numpy()
        pred_out = pred_out.reshape(-1)
        
        # (f) 역 전역 정규화: 예측값에 global_std, global_mean 적용 (np.sinh 제거)
        pred_trans = pred_out * (global_std + eps) + global_mean
        pred_dct = pred_trans
        
        # (g) IDCT를 통해 노이즈 복원
        noise_pred_1d = idct(pred_dct, type=2, norm='ortho')
        noise_pred_list.append(noise_pred_1d)
        
        # (h) 최종 denoised 스펙트럼: 원본 - 예측 노이즈 + baseline + background
        denoised_1d = row_data - noise_pred_1d + baseline + background
        denoise_list.append(denoised_1d)

    print("Baseline: min = {:.4f}, max = {:.4f}".format(np.min(baseline), np.max(baseline)))
    print("Background: min = {:.4f}, max = {:.4f}".format(np.min(background), np.max(background)))
    print("Predicted Noise: min = {:.4f}, max = {:.4f}".format(np.min(noise_pred_1d), np.max(noise_pred_1d)))
    print("Denoised: min = {:.4f}, max = {:.4f}".format(np.min(denoised_1d), np.max(denoised_1d)))

    # (5) 결과 저장 (원본 스펙트럼 포함)
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
