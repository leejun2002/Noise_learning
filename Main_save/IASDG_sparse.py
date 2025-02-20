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

from multiprocessing import Pool
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve

# 모델 파일들 (예: U-Net 계열)
from Model import UNet_CA, UNet, UNet_CA6, UNet_CA7
from Spectra_generator import spectra_generator
from Make_dataset import Read_data, Make_dataset
from config import DefaultConfig

EPS = 1e-6

##########################################
# 0. 희소행렬 기반 ALS 함수
##########################################

def als_baseline_sparse(y, lam=1e5, p=0.01, niter=10):
    """
    희소행렬(sparse matrix) 기반 ALS Baseline Correction.
    기존 als_baseline_numba 대체.

    Args:
        y (1D np.array): 입력 스펙트럼 (길이 L)
        lam (float): 2차 차분 항에 곱할 정칙화 파라미터
        p (float): 비대칭성 파라미터
        niter (int): ALS 반복 횟수
    Returns:
        Z (1D np.array): 추정된 baseline
    """
    L = len(y)
    # 간단한 2차 차분 행렬(대각선 2, -1 위치)
    main_diag = np.full(L, 2.0)
    off_diag = np.full(L - 1, -1.0)
    # 대각행렬 구성
    D2 = diags([main_diag, off_diag, off_diag], [0, -1, 1],
               shape=(L, L), format='csc')
    # lam 배율 곱
    A_regular = lam * D2

    # 초기 가중치 w
    w = np.ones(L, dtype=np.float64)
    Z = np.zeros(L, dtype=np.float64)

    for _ in range(niter):
        W = diags(w, 0, shape=(L, L), format='csc')
        A_sp = A_regular + W   # sparse 행렬
        b = w * y
        # spsolve 로 해 풀기
        Z = spsolve(A_sp, b)
        # w 업데이트
        w = np.where(y > Z, p, 1 - p)

    return Z

##########################################
# 1. 전처리 및 데이터 로딩 (병렬 처리 적용)
##########################################

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

def pipeline_and_save_noise(base_folder, output_mat, config,
                            subfolders=("1", "2", "3"),
                            zero_cut=0.0, row_new_length=1600,
                            col_new_length=640, remove_svs=1):
    """
    1) 텍스트 파일들 읽어 x축 보간 (Row/Column)
    2) ALS baseline correction (희소행렬 이용)
    3) SVD BG 제거
    4) 최종 raw noise(.mat) 저장 (DCT/정규화 없음)
    """
    txt_files = []
    for sf in subfolders:
        sf_path = os.path.join(base_folder, sf)
        fpaths = glob.glob(os.path.join(sf_path, "*.txt"))
        txt_files.extend(fpaths)
    if not txt_files:
        print("No txt files found in subfolders:", subfolders, flush=True)
        return None
    txt_files = sorted(txt_files)

    # 병렬로 row interpolation
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

    # Column Interpolation
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

    # ALS Baseline Correction (컬럼별)
    data_baseline = np.zeros_like(all_spectra_2d)
    total_cols = all_spectra_2d.shape[1]
    for j in range(total_cols):
        baseline = als_baseline_sparse(all_spectra_2d[:, j],
                                       lam=1e5, p=0.01, niter=10)
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

    # SVD Background Removal
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

    # 최종 noise 저장
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
    
    # 2-1. 파이프라인: noise_data.mat 생성
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
    
    # 2-2. 전역 DCT 통계 -> mean, std
    noise_data_dict = sio.loadmat(output_mat, struct_as_record=False, squeeze_me=True)
    raw_noise = noise_data_dict['noise']  # shape: (1600, 640)

    noise_dct = np.zeros_like(raw_noise)
    for j in range(raw_noise.shape[1]):
        noise_dct[:, j] = dct(raw_noise[:, j], type=2, norm='ortho')
    global_mean = np.mean(noise_dct)
    global_std = np.std(noise_dct)
    print(f"Calculated global_mean: {global_mean:.4f}, global_std: {global_std:.4f}", flush=True)

    noise_data_dict.update({"global_mean": global_mean, "global_std": global_std})
    sio.savemat(output_mat, noise_data_dict, do_compression=True)
    print("Updated noise_data.mat with global normalization parameters.", flush=True)
    
    # 2-3. 모델 생성
    model = eval("{}(1,1)".format(config.model_name))
    print(model, flush=True)
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
        model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    save_model_path = save_model_dir(config)
    save_log = save_log_dir(config)
    writer = SummaryWriter(save_log)
    
    # 학습 관련 변수
    global_step = 0  # 여기서부터 시작 (배치마다 +1)
    best_valid_loss = float('inf')

    best_models = []  # (valid_loss, epoch, global_step, checkpoint_file)
    
    # Dataloader 준비
    reader = Read_data(config.train_data_root, config.valid_ratio)
    train_set, valid_set = reader.read_file()
    TrainSet, ValidSet = Make_dataset(train_set), Make_dataset(valid_set)
    train_loader = DataLoader(TrainSet, batch_size=config.batch_size,
                              shuffle=True, pin_memory=True)
    valid_loader = DataLoader(ValidSet, batch_size=256, pin_memory=True)
    
    gen_train = spectra_generator()
    gen_valid = spectra_generator()
    
    # 2-4. 학습 루프
    for epoch in range(config.max_epoch):
        model.train()
        epoch_train_loss = 0.0
        
        for idx, noise in enumerate(train_loader):
            # noise: (batch, spec)
            noise = noise.squeeze().numpy()  # shape: (batch, spec)
            spectra_num, spec = noise.shape
            
            # Clean 생성 -> noisy
            clean_spectra = gen_train.generator(spec, spectra_num).T  # (spec, spectra_num)
            noisy_spectra = clean_spectra + noise  # same shape
            # (spectra_num, spec) => 인덱스 [i, :] 사용 가능
            noisy_spectra = noisy_spectra
            clean_spectra = clean_spectra
            
            # DCT 적용
            input_coef = np.zeros_like(noisy_spectra)
            target_coef = np.zeros_like(noise)
            for i in range(spectra_num):
                input_coef[i, :] = dct(noisy_spectra[i, :], type=2, norm='ortho')
                target_coef[i, :] = dct(noise[i, :], type=2, norm='ortho')
            
            # 정규화
            input_norm = (input_coef - global_mean) / (global_std + EPS)
            target_norm = (target_coef - global_mean) / (global_std + EPS)
            
            # reshape -> (batch, 1, spec)
            input_norm = input_norm.reshape(-1, 1, spec).astype(np.float32)
            target_norm = target_norm.reshape(-1, 1, spec).astype(np.float32)
            
            inp_t = torch.from_numpy(input_norm).to(device)
            out_t = torch.from_numpy(target_norm).to(device)
            
            # Forward
            preds = model(inp_t)
            loss = nn.MSELoss()(preds, out_t)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()

            # global step +1
            global_step += 1
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
        
        # 50 epoch마다 시각화
        if (epoch + 1) % 1 == 0:
            sample_clean = clean_spectra[0, :]
            sample_noisy = noisy_spectra[0, :]
            plt.figure()
            plt.plot(sample_noisy, label="Noisy Spectrum", color="orange")
            plt.plot(sample_clean, label="Clean Spectrum", color="blue")
            plt.title(f"Sample Prediction at Epoch {epoch+1}")
            plt.legend()
            plt.show()
        
        # 상위 3개 모델 저장
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

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

    print("Training finished. Best valid loss =", best_valid_loss)


##########################################
# 3. batch_predict 전처리 (학습 시 global 변수 그대로 사용)
##########################################

def merge_txt_to_single_key_mat_1280(base_dir, out_mat, row_points=1600, target_files=1280):
    """
    base_dir 내부 (하위 폴더 포함) 모든 .txt 파일을 찾은 뒤,
    각 파일의 (x, y) 스펙트럼을 cubic 보간하여 크기가 (target_files, row_points)인
    data_matrix 형태로 저장한다.
    """
    import os
    import glob
    import numpy as np
    from scipy.interpolate import interp1d
    import scipy.io as sio

    # 1) base_dir 아래 모든 .txt 파일을 재귀적으로 수집
    #    -> #1, #2, #3, 그 하위의 0.5M, 0.25M 등 모든 폴더 포함
    file_list = sorted(glob.glob(os.path.join(base_dir, '**', '*.txt'), recursive=True))
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
        
        # 스펙트럼 형식 확인 (2열인지)
        if arr.ndim != 2 or arr.shape[1] < 2:
            print(f"[스킵] {fpath}: shape={arr.shape}", flush=True)
            continue
        
        x = arr[:, 0]
        y = arr[:, 1]
        if x.size < 2:
            print(f"[스킵] {fpath}: 데이터 포인트 부족.", flush=True)
            continue
        
        # 전역 x축 최소/최대
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

    # 2) 전역 x축 [global_min, global_max]를 row_points 개로 등분
    x_new = np.linspace(global_min, global_max, row_points)
    data_matrix_all = np.zeros((n_files, row_points), dtype=np.float32)

    print("[2] 각 파일별 cubic 보간 진행 중...", flush=True)
    idx_valid = 0
    for i, (fpath, x_arr, y_arr) in enumerate(data_entries):
        try:
            f_intp = interp1d(x_arr, y_arr, kind='cubic', fill_value='extrapolate')
            y_interp = f_intp(x_new)
            data_matrix_all[idx_valid, :] = y_interp.astype(np.float32)
            idx_valid += 1
        except Exception as e:
            print(f"Interpolation error in {fpath}: {e}", flush=True)
            continue
    
    # 실제 유효 데이터 개수
    if idx_valid < n_files:
        print(f"[경고] 일부 파일에서 보간 실패. 유효 스펙트럼 개수: {idx_valid}", flush=True)
    else:
        print(f"[확인] 모든 파일 보간 완료. 유효 스펙트럼 개수: {idx_valid}", flush=True)

    # 필요시: idx_valid < n_files 이라면 data_matrix_all을 잘라내기
    data_matrix_all = data_matrix_all[:idx_valid, :]

    # 3) target_files보다 많거나 적을 때 처리
    if idx_valid == target_files:
        data_matrix = data_matrix_all
        print(f"파일 수 {idx_valid} == target_files ({target_files}).", flush=True)
    elif idx_valid > target_files:
        print(f"파일 수 {idx_valid} > {target_files}. 앞쪽 {target_files}개만 사용.", flush=True)
        data_matrix = data_matrix_all[:target_files, :]
    else:
        diff = target_files - idx_valid
        print(f"파일 수 {idx_valid} < {target_files}. 마지막 스펙트럼을 {diff}번 복제합니다.", flush=True)
        data_matrix = np.zeros((target_files, row_points), dtype=np.float32)
        data_matrix[:idx_valid, :] = data_matrix_all
        # 마지막 유효 스펙트럼
        if idx_valid > 0:
            last_spec = data_matrix_all[idx_valid - 1, :]
            for i in range(diff):
                data_matrix[idx_valid + i, :] = last_spec
        else:
            print("[오류] 유효 스펙트럼이 0개입니다.", flush=True)
            return None

    print(f"[결과] 최종 data_matrix shape: {data_matrix.shape} (target_files x {row_points})", flush=True)

    # 4) mat 파일로 저장
    mat_dict = {"data_matrix": data_matrix}
    out_dir = os.path.dirname(out_mat)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    sio.savemat(out_mat, mat_dict)
    print(f"[완료] '{out_mat}'에 data_matrix 저장됨.", flush=True)

    return out_mat


def batch_predict(config):
    """
    예측 시에도 train 때와 동일한 전처리 순서:
     (1) merge_txt_to_single_key_mat_1280 => row_interpolation(+복제)
     (2) column_interpolation
     (3) 열별 ALS
     (4) 2D SVD (remove_svs, fade_factor 동일)
     (5) DCT + 정규화 -> 모델 추론 -> noise 예측 -> denoised
    """
    import os, sys
    import glob
    import numpy as np
    import torch
    import scipy.io as sio
    from scipy.fftpack import dct, idct
    from scipy.interpolate import interp1d
    from scipy.linalg import svd
    import matplotlib.pyplot as plt

    print("batch_predict with unify approach (2D) ...", flush=True)

    # ---------------------------------------------
    # 1) merge_txt_to_single_key_mat_1280 호출
    #    -> row_interpolation 후 data_matrix.mat에 저장
    merged_mat = os.path.join(config.batch_predict_root, "merged_spectra.mat")
    # row_points=1600, target_files=1280 등은 config에서 가져온다고 가정
    out_mat = merge_txt_to_single_key_mat_1280(
        base_dir=config.raw_predict_base,
        out_mat=merged_mat,
        row_points=1600,
        target_files=1280
    )

    if out_mat is None or (not os.path.exists(out_mat)):
        print("[오류] merge_txt_to_single_key_mat_1280 실패.", flush=True)
        return

    # mat 파일에서 data_matrix 읽기
    tmp = sio.loadmat(out_mat, struct_as_record=False, squeeze_me=True)
    if 'data_matrix' not in tmp:
        print("[오류] data_matrix 키가 없음", flush=True)
        return
    # 이 시점에서 data_matrix shape = (1280, 1600)
    #  => (파일개수, row_points)
    data_matrix = tmp['data_matrix'].astype(np.float64)

    # 사용자에 따라 (n_files, spec_len) = (1280, 1600) 일 수도
    n_files, row_points = data_matrix.shape
    print(f"[정보] data_matrix from merge_txt: shape=({n_files}, {row_points})")

    # ---------------------------------------------
    # 2) column interpolation
    #    train에서 column_interpolation은 (row_points × col_new_length)로 만들었으므로
    #    여기서는 row_points=1600, col_new_length=640이 동일한가정
    row_new_length = 1600     # alias
    col_new_length = 640
    # data_matrix: (n_files, row_points) -> 전치해서 (row_points, n_files)
    #    => row_points=1600 => axis=0
    #    => n_files=1280 => axis=1
    # 전치: shape -> (1600, 1280)
    spectra_2d = data_matrix.T  # (1600, 1280)

    print("After row_interpolation (from .mat):", spectra_2d.shape)

    # column interpolation (train과 동일)
    num_rows, num_cols = spectra_2d.shape  # (1600, 1280)
    col_idx = np.arange(num_cols)
    col_new = np.linspace(0, num_cols - 1, col_new_length)  # 0..(1280-1)->640pt
    spectra_2d_col = np.zeros((num_rows, col_new_length), dtype=spectra_2d.dtype)

    for i in range(num_rows):
        f_c = interp1d(col_idx, spectra_2d[i, :], kind='linear')
        spectra_2d_col[i, :] = f_c(col_new)

    print("After column interpolation:", spectra_2d_col.shape)
    if config.visualize_steps:
        plt.figure("Predict: After Column Interpolation")
        plt.imshow(spectra_2d_col, aspect='auto', cmap='jet')
        plt.title("Predict: Column Interpolation")
        plt.colorbar()
        plt.show()

    # ---------------------------------------------
    # 3) 열별 ALS (희소행렬 버전 or 기존)
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    def als_baseline_sparse(y, lam=1e5, p=0.01, niter=10):
        L = len(y)
        main_diag = np.full(L, 2.0)
        off_diag = np.full(L - 1, -1.0)
        D2 = diags([main_diag, off_diag, off_diag], [0, -1, 1], shape=(L, L), format='csc')
        A_reg = lam * D2
        w = np.ones(L, dtype=np.float64)
        Z = np.zeros(L, dtype=np.float64)
        for _ in range(niter):
            from scipy.sparse import csc_matrix
            W = diags(w, 0, shape=(L, L), format='csc')
            A_sp = A_reg + W
            b = w*y
            Z = spsolve(A_sp, b)
            w = np.where(y>Z, p, 1-p)
        return Z

    data_baseline = np.zeros_like(spectra_2d_col)
    for j in range(spectra_2d_col.shape[1]):
        baseline = als_baseline_sparse(spectra_2d_col[:, j], lam=1e5, p=0.01, niter=10)
        data_baseline[:, j] = baseline
    data_bc_removed = spectra_2d_col - data_baseline
    print("After ALS baseline correction:", data_bc_removed.shape)
    if config.visualize_steps:
        plt.figure("Predict: ALS Removed")
        plt.imshow(data_bc_removed, aspect='auto', cmap='jet')
        plt.title("Predict: ALS Correction")
        plt.colorbar()
        plt.show()

    # ---------------------------------------------
    # 4) 2D SVD => remove_svs=1, fade_factor=config.fade_factor
    remove_svs = 1
    fade_factor = config.fade_factor  # train 때와 동일
    U, s, Vt = svd(data_bc_removed, full_matrices=False)
    s_mod = s.copy()
    for i in range(remove_svs):
        if i < len(s_mod):
            s_mod[i] *= fade_factor
    data_bg_removed = U @ np.diag(s_mod) @ Vt
    print("After SVD BG Removal:", data_bg_removed.shape)
    if config.visualize_steps:
        plt.figure("Predict: SVD BG Removal")
        plt.imshow(data_bg_removed, aspect='auto', cmap='jet')
        plt.title("Predict: SVD BG Removal")
        plt.colorbar()
        plt.show()

    # ---------------------------------------------
    # => data_bg_removed가 최종 "Noisy" 스펙트럼
    #    train 파이프라인의 noise_data와 유사한 상태라고 볼 수 있음.
    #    shape=(1600, 640)

    # 5) 딥러닝 모델 로드
    train_noise_mat = os.path.join(config.train_data_root, "noise_data.mat")
    if not os.path.exists(train_noise_mat):
        print("No noise_data.mat found => cannot load global_mean/std.")
        return
    train_mat = sio.loadmat(train_noise_mat, struct_as_record=False, squeeze_me=True)
    if 'global_mean' not in train_mat or 'global_std' not in train_mat:
        print("noise_data.mat has no global_mean / global_std.")
        return
    global_mean = float(train_mat['global_mean'])
    global_std  = float(train_mat['global_std'])

    model_file = os.path.join(config.test_model_dir, f"{config.global_step}.pt")
    if not os.path.exists(model_file):
        print(f"Model checkpoint not found: {model_file}")
        return
    state = torch.load(model_file, map_location='cpu')
    model = eval("{}(1,1)".format(config.model_name))
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # ---------------------------------------------
    # 6) (Row-wise) DCT + 정규화 -> 모델 -> Noise 예측
    nrows, ncols = data_bg_removed.shape  # e.g. (1600, 640)
    predicted_noise = np.zeros_like(data_bg_removed)

    for r in range(nrows):
        row_spectrum = data_bg_removed[r, :]  # shape=(640,)
        resid_dct = dct(row_spectrum, type=2, norm='ortho')
        norm_val = (resid_dct - global_mean) / (global_std + EPS)

        inp_t = torch.from_numpy(norm_val.reshape(1,1,-1).astype(np.float32))
        if torch.cuda.is_available():
            inp_t = inp_t.cuda()
        with torch.no_grad():
            pred_out = model(inp_t).cpu().numpy().reshape(-1)

        # 역정규화 + iDCT
        pred_trans = pred_out*(global_std + EPS) + global_mean
        row_noise  = idct(pred_trans, type=2, norm='ortho')
        predicted_noise[r, :] = row_noise

    # 7) 최종 denoised = data_bg_removed - predicted_noise
    denoised = data_bg_removed - predicted_noise

    # ---------------------------------------------
    # 결과 저장
    out_dir = config.batch_save_root
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = os.path.join(out_dir, "batch_predict_2d_result.mat")

    out_dict = {
        'raw_spectra_2d': data_bg_removed,
        'predicted_noise_2d': predicted_noise,
        'denoised_2d': denoised
    }
    sio.savemat(out_name, out_dict, do_compression=True)
    print(f"[완료] batch_predict => {out_name}")


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
    opt.noise_scale = 1.0
    main(opt)
    plt.show()
