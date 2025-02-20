import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Model import UNet_CA, UNet, UNet_CA6, UNet_CA7
from Spectra_generator import spectra_generator
from Make_dataset import Read_data, Make_dataset
from config import DefaultConfig
import numpy as np
import os
from scipy.fftpack import dct, idct
import scipy.io as sio
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'


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

# 학습 함수
def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = eval("{}(1,1)".format(config.model_name))
    print(model)
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
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
        model_file = os.path.join(save_model_path, str(global_step) + '.pt')
        kwargs = {'map_location': lambda storage, loc: storage.cuda([0, 1])}
        state = torch.load(model_file, **kwargs)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print('Successfully loaded the pretrained model at global step =', global_step)

    reader = Read_data(config.train_data_root, config.valid_ratio)
    train_set, valid_set = reader.read_file()
    TrainSet, ValidSet = Make_dataset(train_set), Make_dataset(valid_set)
    train_loader = DataLoader(TrainSet, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(ValidSet, batch_size=256, pin_memory=True)

    gen_train = spectra_generator()
    gen_valid = spectra_generator()

    writer = SummaryWriter(save_log)

    for epoch in range(config.max_epoch):
        # ========== Training Loop ==========
        for idx, noise in enumerate(train_loader):
            noise = np.squeeze(noise.numpy())
            spectra_num, spec = noise.shape
            clean_spectra = gen_train.generator(spec, spectra_num).T
            noisy_spectra = clean_spectra + noise

            input_coef = np.zeros_like(noisy_spectra)
            output_coef = np.zeros_like(noise)

            for index in range(spectra_num):
                input_coef[index, :] = dct(noisy_spectra[index, :], norm='ortho')
                output_coef[index, :] = dct(noise[index, :], norm='ortho')

            input_coef = input_coef.reshape(-1, 1, spec)
            output_coef = output_coef.reshape(-1, 1, spec)

            input_coef = torch.from_numpy(input_coef).float()
            output_coef = torch.from_numpy(output_coef).float()
            if torch.cuda.is_available():
                input_coef, output_coef = input_coef.cuda(), output_coef.cuda()
                input_coef.to(device)
                output_coef.to(device)

            global_step += 1
            model.train()
            optimizer.zero_grad()

            preds = model(input_coef)
            train_loss = criterion(preds, output_coef)
            train_loss.backward()
            optimizer.step()

            writer.add_scalar('train loss', train_loss.item(), global_step)

            if idx % config.print_freq == 0:
                print(f'epoch {epoch}, batch {idx}, global_step {global_step}, train loss = {train_loss.item()}')

        # ========== Validation Loop ==========
        model.eval()
        valid_loss = 0
        for idx_v, noise in enumerate(valid_loader):
            noise = np.squeeze(noise.numpy())
            spectra_num, spec = noise.shape
            clean_spectra = gen_valid.generator(spec, spectra_num).T
            noisy_spectra = clean_spectra + noise

            input_coef = np.zeros_like(noisy_spectra)
            output_coef = np.zeros_like(noise)

            for index in range(spectra_num):
                input_coef[index, :] = dct(noisy_spectra[index, :], norm='ortho')
                output_coef[index, :] = dct(noise[index, :], norm='ortho')

            input_coef = input_coef.reshape(-1, 1, spec)
            output_coef = output_coef.reshape(-1, 1, spec)

            input_coef = torch.from_numpy(input_coef).float()
            output_coef = torch.from_numpy(output_coef).float()
            if torch.cuda.is_available():
                input_coef, output_coef = input_coef.cuda(), output_coef.cuda()
                input_coef.to(device)
                output_coef.to(device)

            preds_v = model(input_coef)
            valid_loss += criterion(preds_v, output_coef).item()

        valid_loss /= len(valid_loader)
        writer.add_scalar('valid loss', valid_loss, global_step)

        # 학습률 스케줄러 업데이트
        schedual.step()

        # ========== 100 에포크마다 모델 저장 ==========
        if (epoch + 1) % 100 == 0:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_step': global_step,
                'loss': train_loss.item()
            }
            # epoch 정보를 파일명에 포함
            model_file = os.path.join(save_model_path, f'{global_step}.pt')
            torch.save(state, model_file)
            print(f"Model saved at {model_file} (epoch {epoch+1})")


# -------------------------------
# Robust PCA (IALM) 일반 (m x n) 버전
# -------------------------------
def rpca(M, lambda_val=None, tol=1e-7, max_iter=1000):
    """
    Inexact Augmented Lagrange Multiplier(IALM) 로 RPCA
    M: shape (m, n) : m개의 샘플, n차원(스펙트럼 길이)
    M = L + S
    """
    import math
    m_, n_ = M.shape
    norm_M = np.linalg.norm(M, ord='fro')
    if lambda_val is None:
        lambda_val = 1.0 / math.sqrt(max(m_, n_))

    Y = M / max(np.linalg.norm(M, 2), np.linalg.norm(M, np.inf)/lambda_val)
    mu = 1.25 / np.linalg.norm(M, 2)
    mu_bar = mu * 1e7
    rho = 1.5

    L = np.zeros_like(M)
    S = np.zeros_like(M)
    for itr in range(max_iter):
        # SVD -> L 업데이트
        U, sigma, Vt = np.linalg.svd(M - S + (1/mu)*Y, full_matrices=False)
        sigma_thresh = np.maximum(sigma - 1/mu, 0)
        rank = np.sum(sigma_thresh > 0)
        L = (U[:, :rank] * sigma_thresh[:rank]) @ Vt[:rank, :]
        # S 업데이트(soft threshold)
        temp = M - L + (1/mu)*Y
        S = np.sign(temp) * np.maximum(np.abs(temp) - lambda_val/mu, 0)

        Z = M - L - S
        Y = Y + mu*Z
        mu = min(mu*rho, mu_bar)
        err = np.linalg.norm(Z, 'fro') / norm_M
        if err < tol:
            print(f"[RPCA] Converged at iter={itr+1}, err={err:.2e}")
            break
    return L, S

def batch_save_root(config):
    """
    테스트 결과 저장 디렉토리를 구성하여 반환
    """
    result_dir = os.path.join(config.batch_save_root, config.Instrument,
                              config.model_name, f"step_{config.global_step}")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


def batch_predict(config):
    """
    - 1) raman_dataset.mat 로드 (data_matrix, x_new, file_list)
    - 2) 각 행(i)에 대해:
         (a) robust PCA -> (1, spec_len) => L + S
         (b) S -> DCT -> 모델 예측 -> IDCT
         (c) 최종 (L + S_pred) 복원
    - 3) background(L)와 denoised 결과를 .mat에 저장
    """
    print('batch predicting...')

    # -----------------------------
    # (1) 모델 로드
    # -----------------------------
    model = eval("{}(1,1)".format(config.model_name))
    model_file = os.path.join(config.test_model_dir, str(config.global_step) + '.pt')
    if not os.path.exists(model_file):
        print(f"[오류] 모델 파일이 없음: {model_file}")
        return

    state = torch.load(model_file, map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    print(f"Successfully loaded model at global step={state['global_step']}")
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model = model.cuda()
        print("using gpu...")

    # -----------------------------
    # (2) raman_dataset.mat 로드
    # -----------------------------
    mat_filename = os.path.join(config.batch_predict_root, "3siwafer_2d.mat")
    if not os.path.exists(mat_filename):
        print(f"[오류] 파일이 존재하지 않습니다: {mat_filename}")
        return

    tmp = sio.loadmat(mat_filename, struct_as_record=False, squeeze_me=True)

    # data_matrix: shape=(n_files, spec_len)
    if 'data_matrix' not in tmp:
        print("[오류] raman_dataset.mat 에 'data_matrix' 키가 없음.")
        return

    data_matrix = tmp['data_matrix']  # (n_files, spec_len)
    file_list   = tmp['file_list'] if 'file_list' in tmp else None
    x_new       = tmp['x_new'] if 'x_new' in tmp else None

    # shape 확인
    if data_matrix.ndim != 2:
        print(f"[오류] data_matrix shape={data_matrix.shape}, 2D 필요.")
        return

    n_files, spec_len = data_matrix.shape
    print(f"[정보] data_matrix shape=({n_files},{spec_len})")

    # -----------------------------
    # (3) 각 스펙트럼 행(i)을 처리
    # -----------------------------
    from scipy.fftpack import dct, idct

    L_list = []
    S_list = []
    final_list = []

    for i in range(n_files):
        # shape=(spec_len,)
        y = data_matrix[i,:]
        y_2d = y.reshape(1, -1)  # (1, spec_len)

        # RPCA -> L + S
        L, S = rpca(y_2d, tol=1e-6, max_iter=500)
        L_1d = L[0,:]
        S_1d = S[0,:]

        # S -> DCT -> 모델 -> IDCT
        S_dct = dct(S_1d, norm='ortho')
        inp = S_dct.reshape(1,1,-1)
        inp_t = torch.from_numpy(inp).float()
        if torch.cuda.is_available():
            inp_t = inp_t.cuda()
        with torch.no_grad():
            pred_out = model(inp_t).cpu().numpy()  # (1,1,spec_len)
        pred_out = np.squeeze(pred_out)  # (spec_len,)

        S_pred = idct(pred_out, norm='ortho')
        final  = L_1d + S_pred

        L_list.append(L_1d)
        S_list.append(S_pred)
        final_list.append(final)

    # -----------------------------
    # (4) 결과 저장 (.mat)
    # -----------------------------
    L_list_np = np.array(L_list, dtype=object)
    S_list_np = np.array(S_list, dtype=object)
    final_np  = np.array(final_list, dtype=object)

    tmp['background_list'] = L_list_np
    tmp['residual_list']   = S_list_np
    tmp['denoise_list']    = final_np

    out_dir = batch_save_root(config)
    out_name = os.path.join(out_dir, "3siwafer_2d_result.mat")
    sio.savemat(out_name, tmp, do_compression=True)

    print(f"[완료] 최종 결과 저장 -> {out_name}")


def main(config):
    check_dir(config)
    if config.is_training:
        train(config)
    if config.is_batch_predicting:
        batch_predict(config)


if __name__ == '__main__':
    opt = DefaultConfig()
    main(opt)
