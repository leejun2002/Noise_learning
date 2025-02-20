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

from Model import UNet_CA, UNet, UNet_CA6, UNet_CA7
from Spectra_generator import spectra_generator
from Make_dataset import Read_data, Make_dataset
from config import DefaultConfig



# -------------------------------
# 1) Noise 전처리(Interpolate -> SVD->DCT->Transpose->z-score->.mat) 함수
# -------------------------------

def pipeline_and_save_noise(
    base_folder,
    output_mat,
    config,
    subfolders=("1", "2", "3"),
    zero_cut=0.0,
    row_new_length=1600,
    col_new_length=640,
    remove_svs=1
):
    """
    1) base_folder + subfolders 내의 txt 파일을 검색 & (x,y) 로드
    2) row_new_length로 cubic 보간 -> (row_new_length, num_files)
    3) col_new_length로 선형 보간 -> (row_new_length, col_new_length)
    4) SVD 배경 제거(remove_svs개)
    5) 열별 DCT
    6) Transpose( (col_new_length, row_new_length) )
    7) z-score 정규화 & .mat 저장 (key='noise', 'train_mean', 'train_std')
    """

    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.linalg import svd
    from scipy.fftpack import dct

    # 1) txt 파일 로드 & 보간
    txt_files = []
    for sf in subfolders:
        sf_path = os.path.join(base_folder, sf)
        fpaths = glob.glob(os.path.join(sf_path, "*.txt"))
        txt_files.extend(fpaths)

    if not txt_files:
        print("No txt files found:", subfolders)
        return None

    txt_files = sorted(txt_files)
    spectra_list = []

    for fpath in txt_files:
        try:
            arr = np.loadtxt(fpath)
        except Exception as e:
            print(f"[로드 오류] {fpath}: {e}")
            continue

        if arr.ndim != 2 or arr.shape[1] < 2:
            print(f"[스킵] {fpath}, shape={arr.shape}")
            continue

        x_raw, y_raw = arr[:, 0], arr[:, 1]
        # zero_cut 적용
        mask = (x_raw >= zero_cut)
        x_cut = x_raw[mask]
        y_cut = y_raw[mask]
        if x_cut.size < 2:
            print(f"[스킵] {fpath}: 데이터 부족")
            continue

        x_min, x_max = x_cut.min(), x_cut.max()
        if x_min == x_max:
            continue

        # row_new_length 포인트로 cubic 보간
        x_new = np.linspace(x_min, x_max, row_new_length)
        try:
            f_intp = interp1d(x_cut, y_cut, kind='cubic')
            y_new = f_intp(x_new)
        except Exception as e:
            print("[보간 오류]", fpath, e)
            continue

        spectra_list.append(y_new)

    if not spectra_list:
        print("No valid spectra after processing.")
        return None

    # -> (row_new_length, num_files)
    all_spectra = np.array(spectra_list).T
    print("After row interp:", all_spectra.shape)  # (row_new_length, num_files)

    # 열 방향 보간 -> (row_new_length, col_new_length)
    num_rows, num_cols = all_spectra.shape
    col_idx = np.arange(num_cols)
    col_new = np.linspace(0, num_cols - 1, col_new_length)
    all_spectra_2d = np.zeros((num_rows, col_new_length), dtype=all_spectra.dtype)
    for i in range(num_rows):
        f_c = interp1d(col_idx, all_spectra[i, :], kind='linear')
        all_spectra_2d[i, :] = f_c(col_new)

    print("After col interp:", all_spectra_2d.shape)  # (1600, 640)

    # 2) SVD 배경 제거
    U, s, Vt = svd(all_spectra_2d, full_matrices=False)
    s_mod = s.copy()
    for i in range(remove_svs):
        if i < len(s_mod):
            s_mod[i] *= config.fade_factor
    data_bg_removed = U @ np.diag(s_mod) @ Vt
    print("After SVD BG removal:", data_bg_removed.shape)

    # 3) 열별 DCT
    dct_result = np.zeros_like(data_bg_removed)
    for j in range(data_bg_removed.shape[1]):
        dct_result[:, j] = dct(data_bg_removed[:, j], type=2, norm='ortho')
    print("After DCT:", dct_result.shape)

    # 4) Transpose
    noise_data = dct_result.T
    print("After transpose:", noise_data.shape)

    # 5) z-score
    train_mean = np.mean(noise_data)
    train_std  = np.std(noise_data)
    print(f"train_mean={train_mean}, train_std={train_std}")

    if train_std < 1e-6:
        noise_norm = noise_data - train_mean
    else:
        noise_norm = (noise_data - train_mean) / train_std

    print("Normalized: mean =", np.mean(noise_norm), ", std =", np.std(noise_norm))

    # 결과 폴더 생성
    out_dir = os.path.dirname(output_mat)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 저장
    mat_dict = {
        "noise": noise_norm,
        "train_mean": np.array([train_mean]),
        "train_std":  np.array([train_std])
    }
    savemat(output_mat, mat_dict)
    print("Saved noise_data.mat =>", output_mat)

    return output_mat  # 경로 반환


# -------------------------------
# 2) train 함수: 노이즈 전처리 + 모델 학습
# -------------------------------

def train(config):
    """
    1) pipeline_and_save_noise()를 호출하여 noise_data.mat를 생성
    2) 생성된 .mat 파일을 이용하여 Read_data, Make_dataset를 통해 데이터셋 구성
    3) U-Net 모델 학습 진행
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # === (A) 노이즈 데이터 전처리 & .mat 저장 ===
    # config에 전처리 인자들(ex: remove_svs=1, row_new_length=1600, col_new_length=640 등)을 추가하거나
    # 아래에 직접 명시해도 됨.
    output_mat = os.path.join(config.train_data_root, "noise_data.mat")

    print("[INFO] Start pipeline_and_save_noise ...")
    pipeline_and_save_noise(
        base_folder   = config.raw_noise_base,   # 예: /home/jyounglee/NL_real/data/given_data/noise_3siwafer
        output_mat    = output_mat,
        config        = config,
        subfolders    = ("1","2","3"),
        zero_cut      = 0.0,
        row_new_length= 1600,
        col_new_length= 640,
        remove_svs    = 1
    )
    print("[INFO] Noise data .mat 생성 완료 =>", output_mat)

    # === (B) U-Net 모델 생성 ===
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

    # 사전 학습된 모델 로드
    if config.is_pretrain:
        global_step = config.global_step
        model_file = os.path.join(save_model_path, f"{global_step}.pt")
        kwargs = {'map_location': lambda storage, loc: storage.cuda([0, 1])}
        state = torch.load(model_file, **kwargs)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print('Loaded pretrained model at global step =', global_step)

    # === (C) DataLoader 세팅 ===
    #  pipeline_and_save_noise(...)가 만든 noise_data.mat를 읽어오는 경로
    #  config.train_data_root 디렉토리에 noise_data.mat가 있다고 가정.
    reader = Read_data(config.train_data_root, config.valid_ratio)
    train_set, valid_set = reader.read_file()  # 여기서 noise_data.mat 로드
    TrainSet, ValidSet = Make_dataset(train_set), Make_dataset(valid_set)
    train_loader = DataLoader(TrainSet, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(ValidSet, batch_size=256, pin_memory=True)

    gen_train = spectra_generator()
    gen_valid = spectra_generator()
    writer = SummaryWriter(save_log)

    # === (D) Training Loop ===
    for epoch in range(config.max_epoch):
        # --- Training ---
        model.train()
        for idx, noise in enumerate(train_loader):
            noise = noise.squeeze().numpy()  # shape=(batch_size, spec)
            spectra_num, spec = noise.shape

            clean_spectra = gen_train.generator(spec, spectra_num).T
            noisy_spectra = clean_spectra + noise

            input_coef = np.zeros_like(noisy_spectra)
            output_coef = np.zeros_like(noise)
            for i in range(spectra_num):
                input_coef[i, :]  = dct(noisy_spectra[i, :], norm='ortho')
                output_coef[i, :] = dct(noise[i, :], norm='ortho')

            input_coef = input_coef.reshape(-1, 1, spec).astype(np.float32)
            output_coef= output_coef.reshape(-1, 1, spec).astype(np.float32)

            inp_t  = torch.from_numpy(input_coef)
            out_t  = torch.from_numpy(output_coef)
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

        # --- Validation ---
        model.eval()
        valid_loss = 0.0
        for idx_v, noise_v in enumerate(valid_loader):
            noise_v = noise_v.squeeze().numpy()  # shape=(batch_size, spec)
            spectra_num, spec = noise_v.shape

            clean_spectra = gen_valid.generator(spec, spectra_num).T
            noisy_spectra = clean_spectra + noise_v

            input_coef_v = np.zeros_like(noisy_spectra)
            output_coef_v= np.zeros_like(noise_v)
            for i in range(spectra_num):
                input_coef_v[i, :]  = dct(noisy_spectra[i, :], norm='ortho')
                output_coef_v[i, :] = dct(noise_v[i, :], norm='ortho')

            inp_v  = torch.from_numpy(input_coef_v.reshape(-1,1,spec).astype(np.float32))
            out_v  = torch.from_numpy(output_coef_v.reshape(-1,1,spec).astype(np.float32))
            if torch.cuda.is_available():
                inp_v, out_v = inp_v.cuda(), out_v.cuda()

            with torch.no_grad():
                preds_v = model(inp_v)
                v_loss  = criterion(preds_v, out_v)
            valid_loss += v_loss.item()

        valid_loss /= len(valid_loader)
        writer.add_scalar("valid loss", valid_loss, global_step)
        # print(f"[Valid] epoch={epoch}, global_step={global_step}, valid_loss={valid_loss}")

        schedual.step()

        # 100 epoch마다 저장
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




# ------------- batch_predict 텍스트 파일 병합 및 보간 -------------

def merge_txt_to_single_key_mat_1280(
    base_dir="/home/jyounglee/NL_real/data/given_data/3layer_on_siwafer",
    out_mat="/home/jyounglee/NL_real/data/batch_predict/merged_spectra_1280.mat",
    row_points=1600,
    target_files=1280
):
    """
    base_dir 내(하위 폴더 포함)의 모든 .txt 파일을 찾아,
    각 파일의 (x,y) 데이터를 로드 후,
    전역 x_min~x_max 범위에서 row_points (1600)개로 cubic 보간.
    → (n_files, row_points) 배열 생성.
    이후 n_files가 target_files(1280)보다 작으면 부족분은 마지막 스펙트럼을 복제,
    n_files가 많으면 앞쪽 target_files개만 사용하여
    최종 (1280, 1600) 크기의 data_matrix를 key='data_matrix'로 .mat 파일에 저장.
    """
    file_list = sorted(glob.glob(os.path.join(base_dir, "**", "*.txt"), recursive=True))
    if not file_list:
        print(f"[오류] '{base_dir}' 내에 .txt 파일이 없습니다.")
        return None

    # 전역 x 범위 추적
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
    # 보간용 x_new
    x_new = np.linspace(global_min, global_max, row_points)
    data_matrix_all = np.zeros((n_files, row_points), dtype=np.float32)
    print("[2] 각 파일별 cubic 보간 진행 중...")
    for i, (fpath, x_arr, y_arr) in enumerate(data_entries):
        f_intp = interp1d(x_arr, y_arr, kind='cubic', fill_value='extrapolate')
        y_interp = f_intp(x_new)
        data_matrix_all[i, :] = y_interp.astype(np.float32)

    # 파일 수 target_files (1280) 맞추기
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


# ====================================================
# 2) 통합된 batch_predict 함수 (예측 파이프라인)
# ====================================================
def batch_predict(config):
    """
    1) base_folder 내의 모든 txt 파일들을 통합하여 .mat 파일 생성  
       (함수 merge_txt_to_single_key_mat_1280 이용 → shape: (1280,1600))
    2) 저장된 .mat 파일 로드 (key: 'data_matrix')
    3) 각 스펙트럼(행)에 대해:
         (a) SVD로 배경 제거 (rank=1; 필요시 config.rank_bg 사용)
         (b) 행 단위 DCT 적용
         (c) z-score 정규화 (학습 시 저장된 train_mean, train_std 사용)
         (d) 모델 예측 → 예측된 노이즈 DCT 계수
         (e) 역정규화
         (f) IDCT를 통해 노이즈 복원
         (g) 최종 복원: 원본 스펙트럼에서 예측된 노이즈를 제거 (또는 BG 재결합)
    4) 결과를 background_list, noise_pred_list, denoise_list 등에 저장하여 .mat 파일로 출력
    """
    import torch
    import numpy as np
    import scipy.io as sio
    from scipy.fftpack import dct, idct
    from scipy.linalg import svd

    print("batch predicting...")

    # --- (1) txt 파일들을 통합하여 .mat 파일 생성 ---
    merged_mat = os.path.join(config.batch_predict_root, "merged_spectra_1280.mat")
    merge_txt_to_single_key_mat_1280(
        base_dir=config.raw_predict_base,  # 예: 사물 스펙트럼 txt 파일들이 있는 폴더
        out_mat=merged_mat,
        row_points=1600,
        target_files=1280
    )

    # --- (2) 학습 시 사용한 noise 정규화 변수 로드 ---
    if not hasattr(config, 'noise_mat_path'):
        print("[경고] config에 noise_mat_path가 정의되지 않음. (train_mean, train_std 기본값 사용)")
        train_mean, train_std = 0.0, 1.0
    else:
        if not os.path.exists(config.noise_mat_path):
            print(f"[오류] noise_mat_path 파일 없음: {config.noise_mat_path}")
            return
        tmp_meanstd = sio.loadmat(config.noise_mat_path)
        train_mean = float(tmp_meanstd['train_mean'].ravel()[0])
        train_std  = float(tmp_meanstd['train_std'].ravel()[0])
    print(f"[info] Loaded train_mean={train_mean:.4f}, train_std={train_std:.4f}")

    # --- (3) 저장된 .mat 파일 로드 ---
    if not os.path.exists(merged_mat):
        print(f"[오류] merged_mat 파일 없음: {merged_mat}")
        return
    tmp = sio.loadmat(merged_mat, struct_as_record=False, squeeze_me=True)
    if 'data_matrix' not in tmp:
        print("[오류] 'data_matrix' 키가 없음.")
        return
    data_matrix = tmp['data_matrix']  # shape: (1280, 1600)
    data_matrix = data_matrix.astype(np.float64)
    n_files, spec_len = data_matrix.shape
    print(f"[정보] data_matrix shape: ({n_files},{spec_len})")

    # --- (4) 모델 로드 ---
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

    # --- (5) 각 스펙트럼 처리 ---
    background_list = []
    noise_pred_list = []
    denoise_list    = []
    rank_bg = getattr(config, 'rank_bg', 1)  # 기본값 1
    print(f"[info] 배경 제거 랭크: rank_bg={rank_bg}")

    for i in range(n_files):
        # 원본 스펙트럼: shape (1600,)
        row_data = data_matrix[i, :]

        # (a) SVD로 배경 추출 및 residual 계산
        row_2d = row_data.reshape(1, -1)  # (1,1600)

        # SVD를 통한 배경 제거 (fade factor 적용)
        U, s, Vt = svd(row_2d, full_matrices=False)
        bg_mod = np.zeros_like(row_2d)
        for r in range(rank_bg):
            if r < len(s):
                # 첫 singular component에 대해 fade factor를 적용
                if r == 0:
                    s_mod = s[r] * config.fade_factor
                else:
                    s_mod = s[r]  # 나머지는 그대로 사용하거나, 필요에 따라 조정
                bg_mod += np.outer(U[:, r], s_mod * Vt[r, :])
        bg_1d = bg_mod.flatten()
        residual_1d = row_data - bg_1d

        # (b) DCT (행 단위)
        resid_dct = dct(residual_1d, norm='ortho')  # (1600,)

        # [추가] z-score 정규화 직전, residual DCT 계수의 평균과 표준편차 출력
        # current_mean = np.mean(resid_dct)
        # current_std  = np.std(resid_dct)
        # print(f"[DEBUG] Spectrum {i}: DCT mean = {current_mean:.20f}, std = {current_std:.f}")
        
        # resid_dct_norm = (resid_dct - current_mean) / current_std
        # (c) z-score 정규화 (학습 때 구한 train_mean, train_std 사용)
        resid_dct_norm = (resid_dct - train_mean) / train_std

        # (d) 모델 예측: 입력 shape (1,1,1600)
        inp_t = resid_dct_norm.reshape(1, 1, -1)
        inp_torch = torch.from_numpy(inp_t).float()
        if torch.cuda.is_available():
            inp_torch = inp_torch.cuda()
        with torch.no_grad():
            pred_out = model(inp_torch).cpu().numpy()  # (1,1,1600)
        pred_out = pred_out.reshape(-1)  # (1600,)

        # (e) 역정규화: recover DCT 계수
        
        # pred_dct = pred_out * current_std + current_mean
        pred_dct = pred_out * train_std + train_mean

        # (f) IDCT: 예측 노이즈 복원
        noise_pred_1d = idct(pred_dct, norm='ortho')

        # (g) 최종 denoise: (원본 스펙트럼 - 예측된 노이즈)
        denoised_1d = row_data - noise_pred_1d

        background_list.append(bg_1d)
        noise_pred_list.append(noise_pred_1d)
        denoise_list.append(denoised_1d)

    # --- (6) 결과 저장 ---
    # 결과를 기존 .mat 파일(tmp)에 추가
    tmp['background_list'] = np.array(background_list, dtype=object)
    tmp['noise_pred_list'] = np.array(noise_pred_list, dtype=object)
    tmp['denoise_list']    = np.array(denoise_list, dtype=object)
    
    out_dir = config.batch_save_root
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = os.path.join(out_dir, "isdtz_result.mat")
    sio.savemat(out_name, tmp, do_compression=True)
    print(f"[완료] 최종 결과 저장 -> {out_name}")



# 체크포인트 디렉토리 확인 및 생성
def check_dir(config):
    if not os.path.exists(config.checkpoint):
        os.mkdir(config.checkpoint)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    if not os.path.exists(config.test_dir):
        os.mkdir(config.test_dir)


# 테스트 결과 저장 경로 정의
def test_result_dir(config):
    result_dir = os.path.join(config.batch_save_root, config.Instrument,
                              config.model_name, "step_" + str(config.global_step))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


# 모델 저장 경로 정의
def save_model_dir(config):
    save_model_dir = os.path.join(config.checkpoint, config.Instrument,
                                  config.model_name, "batch_" + str(config.batch_size))
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    return save_model_dir


# 로그 경로 정의
def save_log_dir(config):
    save_log_dir = os.path.join(config.logs, config.Instrument,
                                config.model_name, "batch_" + str(config.batch_size))
    if not os.path.exists(save_log_dir):
        os.makedirs(save_log_dir)
    return save_log_dir


# ====================================================
# main 함수
# ====================================================

def main(config):
    check_dir(config)
    if config.is_training:
        train(config)
    if config.is_batch_predicting:
        batch_predict(config)


if __name__ == '__main__':
    opt = DefaultConfig()
    main(opt)
