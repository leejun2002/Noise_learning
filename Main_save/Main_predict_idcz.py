import os
import glob
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.linalg import svd
from scipy.fftpack import dct, idct
from torch.utils.tensorboard import SummaryWriter

# 필요한 모델과 기타 모듈은 상위 폴더에 있을 수 있으므로 부모 디렉토리를 sys.path에 추가 (필요 시)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Model import UNet_CA, UNet, UNet_CA6, UNet_CA7
from Spectra_generator import spectra_generator
from Make_dataset import Read_data, Make_dataset
from config import DefaultConfig

matplotlib.use('TkAgg')  # 또는 'Qt5Agg'

# 체크포인트, 로그, 테스트 결과 저장 경로 생성 함수들
def check_dir(config):
    if not os.path.exists(config.checkpoint):
        os.mkdir(config.checkpoint)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    if not os.path.exists(config.test_dir):
        os.mkdir(config.test_dir)

def test_result_dir(config):
    result_dir = os.path.join(config.test_dir, config.Instrument,
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

# 학습 함수 (train, test, batch_predict는 원래 코드 그대로 사용)
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
    if config.is_pretrain:
        global_step = config.global_step
        model_file = os.path.join(save_model_path, str(global_step) + '.pt')
        kwargs = {'map_location': lambda storage, loc: storage.cuda([0, 1])}
        state = torch.load(model_file, **kwargs)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print('Successfully loaded the pretrained model saved at global step = {}'.format(global_step))
    reader = Read_data(config.train_data_root, config.valid_ratio)
    train_set, valid_set = reader.read_file()
    TrainSet, ValidSet = Make_dataset(train_set), Make_dataset(valid_set)
    train_loader = DataLoader(dataset=TrainSet, batch_size=config.batch_size,
                              shuffle=True, pin_memory=True)
    valid_loader = DataLoader(dataset=ValidSet, batch_size=256, pin_memory=True)
    gen_train = spectra_generator()
    gen_valid = spectra_generator()
    writer = SummaryWriter(save_log)
    for epoch in range(config.max_epoch):
        for idx, noise in enumerate(train_loader):
            noise = np.squeeze(noise.numpy())
            spectra_num, spec = noise.shape
            clean_spectra = gen_train.generator(spec, spectra_num)
            clean_spectra = clean_spectra.T
            noisy_spectra = clean_spectra + noise
            input_coef, output_coef = np.zeros(np.shape(noisy_spectra)), np.zeros(np.shape(noise))
            for index in range(spectra_num):
                input_coef[index, :] = dct(noisy_spectra[index, :], norm='ortho')
                output_coef[index, :] = dct(noise[index, :], norm='ortho')
            input_coef = np.reshape(input_coef, (-1, 1, spec))
            output_coef = np.reshape(output_coef, (-1, 1, spec))
            input_coef = torch.from_numpy(input_coef).type(torch.FloatTensor)
            output_coef = torch.from_numpy(output_coef).type(torch.FloatTensor)
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
            writer.add_scalar('train loss', train_loss.item(), global_step=global_step)
            if idx % config.print_freq == 0:
                print('epoch {}, batch {}, global step {}, train loss = {}'.format(
                    epoch, idx, global_step, train_loss.item()))
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'global_step': global_step, 'loss': train_loss.item()}
        model_file = os.path.join(save_model_path, str(global_step) + '.pt')
        torch.save(state, model_file)
        model.eval()
        valid_loss = 0
        for idx_v, noise in enumerate(valid_loader):
            noise = np.squeeze(noise.numpy())
            spectra_num, spec = noise.shape
            clean_spectra = gen_valid.generator(spec, spectra_num)
            clean_spectra = clean_spectra.T
            noisy_spectra = clean_spectra + noise
            input_coef, output_coef = np.zeros(np.shape(noisy_spectra)), np.zeros(np.shape(noise))
            for index in range(spectra_num):
                input_coef[index, :] = dct(noisy_spectra[index, :], norm='ortho')
                output_coef[index, :] = dct(noise[index, :], norm='ortho')
            input_coef = np.reshape(input_coef, (-1, 1, spec))
            output_coef = np.reshape(output_coef, (-1, 1, spec))
            input_coef = torch.from_numpy(input_coef).type(torch.FloatTensor)
            output_coef = torch.from_numpy(output_coef).type(torch.FloatTensor)
            if torch.cuda.is_available():
                input_coef, output_coef = input_coef.cuda(), output_coef.cuda()
                input_coef.to(device)
                output_coef.to(device)
            preds_v = model(input_coef)
            valid_loss += criterion(preds_v, output_coef).item()
        valid_loss = valid_loss / len(valid_loader)
        writer.add_scalar('valid loss', valid_loss, global_step=global_step)
        schedual.step()

def test(config):
    print('testing...')
    model = eval("{}(1,1)".format(config.model_name))
    model_file = os.path.join(config.test_model_dir, str(config.global_step) + '.pt')
    state = torch.load(model_file, weights_oNL_realy=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    print('Successfully loaded The model saved at global step = {}'.format(state['global_step']))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print('using gpu...')
    filenames = os.listdir(config.test_data_root)
    for file in filenames:
        if os.path.splitext(file)[1] == '.mat':
            name = os.path.join(config.test_data_root, file)
            tmp = sio.loadmat(name)
            inpts, inptr = np.array(tmp[config.test_varible[0]]), np.array(tmp[config.test_varible[1]])
            inpts, inptr = inpts.T, inptr.T
            nums, spec = inpts.shape
            numr, _ = inptr.shape
            for idx in range(nums):
                inpts[idx, :] = dct(np.squeeze(inpts[idx, :]), norm='ortho')
            for idx in range(numr):
                inptr[idx, :] = dct(np.squeeze(inptr[idx, :]), norm='ortho')
            inpts = np.array([inpts]).reshape((nums, 1, spec))
            inptr = np.array([inptr]).reshape((numr, 1, spec))
            inpts, inptr = torch.from_numpy(inpts).float(), torch.from_numpy(inptr).float()
            inptt = torch.cat([inpts, inptr], dim=0)
            test_size = 32
            group_total = torch.split(inptt, test_size)
            predt = []
            for i in range(len(group_total)):
                xt = group_total[i]
                if torch.cuda.is_available():
                    xt = xt.cuda()
                yt = model(xt).detach().cpu()
                predt.append(yt)
            predt = torch.cat(predt, dim=0)
            predt = predt.numpy()
            predt = np.squeeze(predt)
            preds, predr = predt[:nums, :], predt[nums:, :]
            for idx in range(nums):
                preds[idx, :] = idct(np.squeeze(preds[idx, :]), norm='ortho')
                predr[idx, :] = idct(np.squeeze(predr[idx, :]), norm='ortho')
            tmp['preds'], tmp['predr'] = preds.T, predr.T
            test_dir = test_result_dir(config)
            base_name, ext = os.path.splitext(file)
            filename = os.path.join(test_dir, f"{base_name}_result{ext}")
            sio.savemat(filename, tmp)


# 수정된 predict 함수: predict 전처리 파이프라인을 적용 (Interpolation → DCT → Clamping → Normalization)
def predict():
    print("predicting...")

    # === 하드코딩 값들 (학습 시 사용한 값과 동일하게 설정) ===
    # 모델 관련
    model_name = "UNet_CA"          # 사용할 모델 이름
    global_step = 10000             # 예시 global step (학습 시 저장된 모델 번호)
    save_model_path = "/home/jyounglee/NL_real/noise_dp6/result/model"  # 저장된 모델 폴더 경로
    model_file = os.path.join(save_model_path, f"{global_step}.pt")

    # 전처리 및 정규화 관련 파라미터
    target_points = 1600            # 스펙트럼 길이(보간 후)
    dct_threshold = 10000           # DCT 변환 후 low-frequency 영역 clamping 임계값
    low_freq_region = 50            # clamping 적용할 low-frequency 영역 (0 ~ 50 인덱스)
    
    # # 학습 시 저장한 train_mean과 train_std (예시 값; 실제 값으로 교체)
    # train_mean = 9.42063274094999
    # train_std  = 7900.101090550069
    
    # 학습 시 저장된 .mat 파일에서 train_mean과 train_std 로드
    noise_mat_path = "/home/jyounglee/NL_real/noise_data_processing/noise_dp6/result/noise_3siwafer_640x1600.mat"
    mat_data = sio.loadmat(noise_mat_path)
    train_mean = float(mat_data["train_mean"].ravel()[0])
    train_std  = float(mat_data["train_std"].ravel()[0])

    # 예측 데이터 관련 경로
    predict_root = "/home/jyounglee/NL_real/predict_processing"            # 입력 txt 파일들이 있는 폴더
    predict_save_root = "/home/jyounglee/NL_real/predict_processing/denoised"  # denoised 결과를 저장할 폴더
    if not os.path.exists(predict_save_root):
        os.makedirs(predict_save_root)

    # === 모델 로드 ===
    model = UNet_CA(1, 1)
    state = torch.load(model_file, map_location="cpu")  # GPU 환경에 맞게 수정 가능
    # 만약 저장된 state에 'model' 키가 있다면:
    model.load_state_dict({k.replace("module.", ""): v for k, v in state["model"].items()})
    print("Successfully loaded the model saved at global step =", state["global_step"])
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("using gpu...")

    # === predict 전처리 및 예측 ===
    filenames = os.listdir(predict_root)
    plt.figure(figsize=(9, 7))
    for file in filenames:
        if os.path.splitext(file)[1] == ".txt":
            file_path = os.path.join(predict_root, file)
            new_name = os.path.join(predict_save_root, "dn_" + file)

            # 1. 텍스트 파일 로드 (탭 구분, 첫 열: Raman Shift, 두번째 열: Intensity)
            data = np.loadtxt(file_path, delimiter="\t").astype(float)
            wave = data[:, 0]
            intensity = data[:, 1]

            # 2. Interpolation: 스펙트럼 길이가 target_points와 다르면 선형 보간 수행
            if len(intensity) != target_points:
                x_orig = np.linspace(wave.min(), wave.max(), len(intensity))
                x_new = np.linspace(wave.min(), wave.max(), target_points)
                f_intp = interp1d(x_orig, intensity, kind="linear")
                intensity = f_intp(x_new)
                wave = x_new

            # 3. DCT 변환 (type=2, norm='ortho')
            dct_coeff = dct(intensity, type=2, norm="ortho")

            # 4. Clamping: low-frequency 영역(0 ~ low_freq_region) clamping 적용
            dct_coeff[:low_freq_region] = np.clip(dct_coeff[:low_freq_region], -dct_threshold, dct_threshold)

            # 5. 정규화: 학습 시의 train_mean과 train_std로 z‑score 정규화
            dct_norm = (dct_coeff - train_mean) / train_std

            # 6. 모델 입력: 정규화된 DCT 계수를 (1,1,-1) 형태로 리쉐이프 후 torch tensor 변환
            input_tensor = dct_norm.reshape(1, 1, -1)
            input_tensor = torch.from_numpy(input_tensor).float()
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()

            # 7. 모델 예측: noise의 정규화된 DCT 계수를 예측
            with torch.no_grad():
                output_tensor = model(input_tensor)
            pred_dct_norm = output_tensor.cpu().numpy().reshape(-1)

            # 8. 역정규화: 예측된 계수를 학습 시 값으로 복원
            pred_dct = pred_dct_norm * train_std + train_mean

            # 9. IDCT: 예측된 DCT 계수를 IDCT로 변환하여 noise 복원
            pred_noise = idct(pred_dct, type=2, norm="ortho")

            # 10. Denoised 스펙트럼 생성: 원본 intensity에서 예측된 noise 제거
            denoised_intensity = intensity - pred_noise

            # 11. 결과 저장: (wave, denoised_intensity)를 txt 파일로 저장 (소수점 이하 12자리)
            result_array = np.column_stack((wave, denoised_intensity))
            np.savetxt(new_name, result_array, delimiter="\t", fmt="%.12f")
            print("Saved denoised spectrum to", new_name)

            # 선택: 결과 플롯 (원한다면)
            plt.plot(wave, intensity, "--", label="Noisy Spectrum")
            plt.plot(wave, denoised_intensity, label="Denoised Spectrum", linewidth=2)
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def batch_predict(config):
    print('batch predicting...')
    model = eval("{}(1,1)".format(config.model_name))
    model_file = os.path.join(config.test_model_dir, str(config.global_step) + '.pt')
    state = torch.load(model_file, weights_oNL_realy=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    print('Successfully loaded The model saved at global step = {}'.format(state['global_step']))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print('using gpu...')
    filenames = os.listdir(config.predict_root)
    for file in filenames:
        if os.path.splitext(file)[1] == '.mat':
            name = os.path.join(config.predict_root, file)
            tmp = sio.loadmat(name)
            inpts = np.array(tmp['cube'])
            inpts = np.array(tmp['spectrum'])
            print(f"Shape of spectrum: {inpts.shape}")
            print(f"Data example: {inpts[:5]}")
            inpts = inpts.T
            nums, spec = inpts.shape
            for idx in range(nums):
                inpts[idx, :] = dct(np.squeeze(inpts[idx, :]), norm='ortho')
            inpts = np.array([inpts]).reshape((nums, 1, spec))
            inpts = torch.from_numpy(inpts).float()
            test_size = 32
            group_total = torch.split(inpts, test_size)
            preds = []
            for i in range(len(group_total)):
                xt = group_total[i]
                if torch.cuda.is_available():
                    xt = xt.cuda()
                yt = model(xt).detach().cpu()
                preds.append(yt)
            preds = torch.cat(preds, dim=0)
            preds = preds.numpy()
            preds = np.squeeze(preds)
            for idx in range(nums):
                preds[idx, :] = idct(np.squeeze(preds[idx, :]), norm='ortho')
            tmp['preds'] = preds.T
            test_dir = test_result_dir(config)
            filename = os.path.join(test_dir, "".join(file))
            sio.savemat(filename, tmp)


def main(config):
    check_dir(config)
    if config.is_training:
        train(config)
    if config.is_testing:
        test(config)
    if config.is_predicting:
        predict(config)
    if config.is_batch_predicting:
        batch_predict(config)

if __name__ == '__main__':
    opt = DefaultConfig()
    main(opt)
