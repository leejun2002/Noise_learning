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

# 모델 파일들 (예: U-Net 계열)
from Model import UNet_CA, UNet, UNet_CA6, UNet_CA7
from Spectra_generator import spectra_generator
from Make_dataset import Read_data, Make_dataset
from config import DefaultConfig
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from multiprocessing import Pool

##########################################
# 1. 전처리 및 데이터 로딩 (병렬 처리 적용)
##########################################

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
    # 2차 미분 행렬 계산 (밀집 행렬 방식)
    D = np.diff(np.eye(L), 2)
    D = lam * D.dot(D.T)
    w = np.ones(L)
    for i in range(niter):
        W = np.diag(w)
        Z = np.linalg.solve(W + D, w * y)
        w = p * (y > Z) + (1 - p) * (y < Z)
    return Z

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

def process_file(fpath, zero_cut, row_new_length):
    """
    개별 텍스트 파일을 읽어 보간된 스펙트럼(1D 벡터)를 반환합니다.
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

def pipeline_and_save_noise(base_folder, output_mat, config, subfolders=("1", "2", "3"),
                              zero_cut=0.0, row_new_length=1600, col_new_length=640, remove_svs=1):
    """
    전체 전처리 파이프라인.
      1) 텍스트 파일들을 병렬 처리하여 행 보간 → all_spectra
      2) 열 보간을 통해 2D 스펙트럼 생성 → all_spectra_2d
      3) 각 컬럼에 대해 ALS baseline correction 적용 → data_bc_removed
      4) SVD를 통해 배경(BG) 제거 → data_bg_removed
      5) 각 열에 대해 DCT 적용 → dct_result
      6) 행렬 전치 및 전역 정규화 → norm_data
      7) 결과를 .mat 파일로 저장
      8) 중간 결과 플롯
    """
    eps = 1e-6

    # Step 1: Row Interpolation (병렬 처리)
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

    # Step 2: Column Interpolation
    num_rows, num_cols = all_spectra.shape
    col_idx = np.arange(num_cols)
    col_new = np.linspace(0, num_cols - 1, col_new_length)
    all_spectra_2d = np.zeros((num_rows, col_new_length), dtype=all_spectra.dtype)
    for i in range(num_rows):
        f_c = interp1d(col_idx, all_spectra[i, :], kind='linear')
        all_spectra_2d[i, :] = f_c(col_new)
    print("After column interpolation:", all_spectra_2d.shape)

    # Step 3: ALS Baseline Correction (컬럼별)
    data_baseline = np.zeros_like(all_spectra_2d)
    for j in range(all_spectra_2d.shape[1]):
        baseline = als_baseline(all_spectra_2d[:, j], lam=1e5, p=0.01, niter=10)
        data_baseline[:, j] = baseline
    data_bc_removed = all_spectra_2d - data_baseline
    print("After ALS baseline correction:", data_bc_removed.shape)

    # Step 4: SVD Background Removal
    U, s, Vt = svd(data_bc_removed, full_matrices=False)
    s_mod = s.copy()
    for i in range(remove_svs):
        if i < len(s_mod):
            s_mod[i] *= config.fade_factor
    data_bg_removed = U @ np.diag(s_mod) @ Vt
    print("After SVD background removal:", data_bg_removed.shape)

    # Step 5: DCT 적용
    dct_result = np.zeros_like(data_bg_removed)
    for j in range(data_bg_removed.shape[1]):
        dct_result[:, j] = dct(data_bg_removed[:, j], type=2, norm='ortho')
    print("After DCT:", dct_result.shape)

    # Step 6: Transpose 및 전역 정규화
    noise_data = dct_result.T  # shape: [col_new_length, row_new_length]
    print("After transpose:", noise_data.shape)
    global_mean = np.mean(noise_data)
    global_std = np.std(noise_data)
    norm_data = (noise_data - global_mean) / (global_std + eps)
    print("After global normalization:", norm_data.shape)
    print("Global mean: {:.4f}, Global std: {:.4f}".format(global_mean, global_std))
    
    # Step 7: 결과 시각화
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

    # Step 8: 결과 저장
    out_dir = os.path.dirname(output_mat)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    savemat(output_mat, {"noise": norm_data,
                         "global_mean": global_mean,
                         "global_std": global_std})
    print("Saved noise_data.mat =>", output_mat)
    return output_mat

##########################################
# 2. 모델 학습 및 최적화 (Adam, ReduceLROnPlateau, Early Stopping, Gradient Clipping)
##########################################

def train(config):
    """
    1) pipeline_and_save_noise()로 noise_data.mat 생성
    2) 생성된 .mat 파일을 이용해 데이터셋 구성
    3) U-Net 모델 학습 진행 (Adam 옵티마이저, ReduceLROnPlateau, Early Stopping, Gradient Clipping 적용)
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # (A) 노이즈 데이터 전처리 및 저장
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
    plt.show()  # 전처리 플롯 표시
    print("[INFO] Noise data .mat 생성 완료 =>", output_mat)
    
    # (B) 모델 생성
    model = eval("{}(1,1)".format(config.model_name))
    print(model)
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
        model.to(device)
    
    # Adam 옵티마이저 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # ReduceLROnPlateau: validation loss 개선에 따라 학습률 감소
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    save_model_path = save_model_dir(config)
    save_log = save_log_dir(config)
    writer = SummaryWriter(save_log)
    
    global_step = 0
    best_valid_loss = float('inf')
    patience = 10  # early stopping patience
    no_improve_epochs = 0

    # (C) DataLoader 구성 (noise_data.mat 이용)
    reader = Read_data(config.train_data_root, config.valid_ratio)
    train_set, valid_set = reader.read_file()
    TrainSet, ValidSet = Make_dataset(train_set), Make_dataset(valid_set)
    train_loader = DataLoader(TrainSet, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(ValidSet, batch_size=256, pin_memory=True)
    
    gen_train = spectra_generator()
    gen_valid = spectra_generator()
    
    # (D) Training Loop
    for epoch in range(config.max_epoch):
        model.train()
        epoch_train_loss = 0.0
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
    
            inp_t = torch.from_numpy(input_coef).to(device)
            out_t = torch.from_numpy(output_coef).to(device)
    
            global_step += 1
            preds = model(inp_t)
            loss = nn.MSELoss()(preds, out_t)
    
            optimizer.zero_grad()
            loss.backward()
            # Gradient Clipping (최대 norm 5)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    
            epoch_train_loss += loss.item()
            if idx % config.print_freq == 0:
                print(f"[Train] epoch={epoch}, batch={idx}, global_step={global_step}, loss={loss.item():.6f}")
            writer.add_scalar("train loss", loss.item(), global_step)
    
        epoch_train_loss /= len(train_loader)
    
        # Validation Loop
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
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
    
                inp_v = torch.from_numpy(input_coef_v.reshape(-1, 1, spec).astype(np.float32)).to(device)
                out_v = torch.from_numpy(output_coef_v.reshape(-1, 1, spec).astype(np.float32)).to(device)
                preds_v = model(inp_v)
                v_loss = nn.MSELoss()(preds_v, out_v)
                valid_loss += v_loss.item()
    
        valid_loss /= len(valid_loader)
        writer.add_scalar("valid loss", valid_loss, global_step)
        # Scheduler: ReduceLROnPlateau step
        scheduler.step(valid_loss)
    
        print(f"[Epoch {epoch}] Train Loss: {epoch_train_loss:.6f} | Valid Loss: {valid_loss:.6f}")
    
        # Early Stopping & Checkpointing: Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            no_improve_epochs = 0
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_step': global_step,
                'loss': loss.item()
            }
            best_model_file = os.path.join(save_model_path, "best_model.pt")
            torch.save(state, best_model_file)
            print(f"[Save] Best model saved at epoch {epoch} with valid loss {valid_loss:.6f}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best valid loss: {best_valid_loss:.6f}")
                break
            
##########################################
# 3. batch_predict 전처리(보간)
##########################################

def merge_txt_to_single_key_mat_1280(base_dir, out_mat, row_points=1600, target_files=1280):
    """
    base_dir 내의 모든 .txt 파일을 검색하여,
    각 파일의 (x, y) 데이터를 로드하고, 전역 x 범위에 대해 cubic 보간을 수행하여
    (n_files, row_points) 배열을 생성합니다.
    파일 수가 target_files보다 적으면 마지막 스펙트럼을 복제하고, 많으면 앞쪽 target_files개만 사용하여
    최종 (target_files, row_points) 크기의 data_matrix를 .mat 파일로 저장합니다.
    """
    import os
    import glob
    import numpy as np
    from scipy.interpolate import interp1d
    import scipy.io as sio

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
        try:
            f_intp = interp1d(x_arr, y_arr, kind='cubic', fill_value='extrapolate')
            y_interp = f_intp(x_new)
            data_matrix_all[i, :] = y_interp.astype(np.float32)
        except Exception as e:
            print(f"Interpolation error in {fpath}: {e}")
            continue
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
    print(f"[결과] 최종 data_matrix shape: {data_matrix.shape} (target_files x {row_points})")
    mat_dict = {"data_matrix": data_matrix}
    out_dir = os.path.dirname(out_mat)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    sio.savemat(out_mat, mat_dict)
    print(f"[완료] '{out_mat}'에 data_matrix 저장됨.")
    return out_mat


##########################################
# 4. 후처리 및 예측 파이프라인 (역정규화, IDCT, 통계 로깅)
##########################################

def batch_predict(config):
    """
    1) 텍스트 파일들을 병합하여 merged_spectra.mat 생성 (merge_txt_to_single_key_mat_1280 이용)
    2) merged_spectra.mat 로부터 data_matrix를 로드
    3) 각 스펙트럼에 대해:
         (a) ALS baseline correction, SVD BG 제거, DCT 수행 → resid_dct
         (b) 전역 정규화 (학습 시 사용한 global_mean, global_std)
         (c) 모델 예측 및 역정규화 후 IDCT 수행 → noise_pred
         (d) 최종 denoised spectrum = 원본 - noise_pred + baseline + background
    4) 결과를 .mat 파일에 저장하고, 전체 통계(평균, 분산, min/max 등)를 출력
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

    # (2) merged_spectra.mat 로드
    if not os.path.exists(merged_mat):
        print(f"[오류] merged_mat 파일 없음: {merged_mat}")
        return
    tmp = sio.loadmat(merged_mat, struct_as_record=False, squeeze_me=True)
    if 'data_matrix' not in tmp:
        print("[오류] 'data_matrix' 키가 없음.")
        return
    data_matrix = tmp['data_matrix'].astype(np.float64)
    n_files, spec_len = data_matrix.shape
    print(f"[정보] data_matrix shape: ({n_files}, {spec_len})")

    # (2.5) 전역 정규화 파라미터 불러오기 (noise_data.mat)
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

    # (4) 각 스펙트럼 처리 및 예측
    background_list = []   # SVD로 제거한 BG 저장
    baseline_list = []     # ALS baseline 저장
    noise_pred_list = []   # 예측된 노이즈 저장
    denoise_list = []      # 최종 denoised spectrum 저장
    eps = 1e-6
    lam = 1e5
    p = 0.01
    niter = 10

    for i in range(n_files):
        row_data = data_matrix[i, :]
        # (a) ALS baseline correction
        baseline = als_baseline(row_data, lam=lam, p=p, niter=niter)
        baseline_list.append(baseline)
        data_bc_removed = row_data - baseline

        # (b) SVD BG 제거
        U, s, Vt = svd(data_bc_removed.reshape(-1, 1), full_matrices=False)
        background = (U * s) @ Vt
        background = background.flatten()
        background_list.append(background)
        data_bg_removed = data_bc_removed - background

        # (c) 행 단위 DCT 적용
        resid_dct = dct(data_bg_removed, type=2, norm='ortho')
        
        # (d) 전역 정규화 (학습 파라미터 사용)
        norm_val = (resid_dct - global_mean) / (global_std + eps)
        
        # (e) 모델 예측
        inp_t = norm_val.reshape(1, 1, -1)
        inp_torch = torch.from_numpy(inp_t).float()
        if torch.cuda.is_available():
            inp_torch = inp_torch.cuda()
        with torch.no_grad():
            pred_out = model(inp_torch).cpu().numpy()
        pred_out = pred_out.reshape(-1)
        
        # (f) 역 정규화 (선형 변환, np.sinh 제거)
        pred_trans = pred_out * (global_std + eps) + global_mean
        pred_dct = pred_trans
        
        # (g) IDCT를 통해 노이즈 복원
        noise_pred_1d = idct(pred_dct, type=2, norm='ortho')
        noise_pred_list.append(noise_pred_1d)
        
        # (h) 최종 denoised: 원본 - noise_pred + baseline + background
        denoised_1d = row_data - noise_pred_1d + baseline + background
        denoise_list.append(denoised_1d)

    # 전체 통계 로깅 (예: 평균, 분산, min, max)
    all_baseline = np.array(baseline_list)
    all_background = np.array(background_list)
    all_pred_noise = np.array(noise_pred_list)
    all_denoised = np.array(denoise_list)
    print("Predicted Noise: mean = {:.4f}, std = {:.4f}, min = {:.4f}, max = {:.4f}".format(
        np.mean(all_pred_noise), np.std(all_pred_noise), np.min(all_pred_noise), np.max(all_pred_noise)))
    print("Denoised: mean = {:.4f}, std = {:.4f}, min = {:.4f}, max = {:.4f}".format(
        np.mean(all_denoised), np.std(all_denoised), np.min(all_denoised), np.max(all_denoised)))
    
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
    main(opt)
    plt.show()
