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
    result_dir = os.path.join(config.test_dir, config.Instrument,
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
    # 사용 GPU 개수 지정
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 모델 생성 --> ** 모델 설정 중요 **
    model = eval("{}(1,1)".format(config.model_name))
    print(model)
    # GPU 사용 설정
    if torch.cuda.is_available():
        # 멀티 GPU 학습 설정
        model = nn.DataParallel(model)
        model = model.cuda()
        model.to(device)
    # 손실 함수 정의
    criterion = nn.MSELoss()
    # 최적화 알고리즘 정의
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    # 학습률 조정 정의
    schedual = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0005)
    # 모델 저장 경로
    save_model_path = save_model_dir(config)
    # 로그 저장 경로
    save_log = save_log_dir(config)
    # 전체 학습 스텝 수
    global_step = 0
    # 사전 학습된 모델 로드 여부
    if config.is_pretrain:
        global_step = config.global_step
        model_file = os.path.join(save_model_path, str(global_step) + '.pt')
        # 모델 파라미터 로드
        kwargs = {'map_location': lambda storage, loc: storage.cuda([0, 1])}
        state = torch.load(model_file, **kwargs)
        # from collections import OrderedDict
        # new_state = OrderedDict()
        # for k, v in state['model'].items():
        # name = 'module.' + k  # add `module.`
        # new_state[name] = v
        # model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
        # model.load_state_dict(new_state)
        model.load_state_dict(state['model'])
        # 옵티마이저 상태 복원
        optimizer.load_state_dict(state['optimizer'])
        print('Successfully loaded the pretrained model saved at global step = {}'.format(global_step))
    # 데이터셋 로드
    reader = Read_data(config.train_data_root, config.valid_ratio)
    train_set, valid_set = reader.read_file()
    # 데이터셋 로더 생성
    TrainSet, ValidSet = Make_dataset(train_set), Make_dataset(valid_set)

    train_loader = DataLoader(dataset=TrainSet, batch_size=config.batch_size,
                              shuffle=True, pin_memory=True)
    valid_loader = DataLoader(dataset=ValidSet, batch_size=256, pin_memory=True)
    # 시뮬레이션 스펙트럼 생성기
    gen_train = spectra_generator()
    gen_valid = spectra_generator()
    # TensorBoard 정의
    writer = SummaryWriter(save_log)
    for epoch in range(config.max_epoch):
        for idx, noise in enumerate(train_loader):
            noise = np.squeeze(noise.numpy())
            spectra_num, spec = noise.shape
            clean_spectra = gen_train.generator(spec, spectra_num)
            # 전치
            clean_spectra = clean_spectra.T
            noisy_spectra = clean_spectra + noise
            # 입력과 출력 정의
            input_coef, output_coef = np.zeros(np.shape(noisy_spectra)), np.zeros(np.shape(noise))
            # 사전 처리 수행, DCT 변환
            for index in range(spectra_num):
                input_coef[index, :] = dct(noisy_spectra[index, :], norm='ortho')
                output_coef[index, :] = dct(noise[index, :], norm='ortho')
            # 3차원으로 리쉐이프
            input_coef = np.reshape(input_coef, (-1, 1, spec))
            output_coef = np.reshape(output_coef, (-1, 1, spec))
            # 텐서로 변환
            input_coef = torch.from_numpy(input_coef)
            output_coef = torch.from_numpy(output_coef)
            # float 타입으로 변환
            input_coef = input_coef.type(torch.FloatTensor)
            output_coef = output_coef.type(torch.FloatTensor)
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
            # 학습 손실과 검증 손실을 로그에 기록
            writer.add_scalar('train loss', train_loss.item(), global_step=global_step)
            # 설정된 주기마다 학습 손실 출력
            if idx % config.print_freq == 0:
                print('epoch {}, batch {}, global step  {}, train loss = {}'.format(
                    epoch, idx, global_step, train_loss.item()))

        # 모델 파라미터 저장
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'global_step': global_step, 'loss': train_loss.item()}
        model_file = os.path.join(save_model_path, str(global_step) + '.pt')
        torch.save(state, model_file)
        """
        로드 시 다음 코드를 사용
        state = torch.load(save_model_path)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        global_step = state['global_step']
        """
        # 검증: BN(Batch Normalization) 평가를 위해 eval() 호출
        model.eval()
        valid_loss = 0
        for idx_v, noise in enumerate(valid_loader):
            noise = np.squeeze(noise.numpy())
            spectra_num, spec = noise.shape
            # print(noise.shape)
            clean_spectra = gen_valid.generator(spec, spectra_num)
            # 전치
            clean_spectra = clean_spectra.T
            noisy_spectra = clean_spectra + noise
            # 입출력 정의
            input_coef, output_coef = np.zeros(np.shape(noisy_spectra)), np.zeros(np.shape(noise))
            # 전처리 수행, DCT 변환
            for index in range(spectra_num):
                input_coef[index, :] = dct(noisy_spectra[index, :], norm='ortho')
                output_coef[index, :] = dct(noise[index, :], norm='ortho')
            # 3차원으로 변환
            input_coef = np.reshape(input_coef, (-1, 1, spec))
            output_coef = np.reshape(output_coef, (-1, 1, spec))
            # tensor로 변환
            input_coef = torch.from_numpy(input_coef)
            output_coef = torch.from_numpy(output_coef)
            # float tensor
            input_coef = input_coef.type(torch.FloatTensor)
            output_coef = output_coef.type(torch.FloatTensor)
            if torch.cuda.is_available():
                input_coef, output_coef = input_coef.cuda(), output_coef.cuda()
                input_coef.to(device)
                output_coef.to(device)
            preds_v = model(input_coef)
            valid_loss += criterion(preds_v, output_coef).item()
        valid_loss = valid_loss / len(valid_loader)
        writer.add_scalar('valid loss', valid_loss, global_step=global_step)
        # 한 에포크 완료 후 학습률 조정
        schedual.step()


def test(config):
    print('testing...')
    # 모델 생성
    model = eval("{}(1,1)".format(config.model_name))
    # 모델 파일 경로 생성
    # save_model_path = save_model_dir(config)
    model_file = os.path.join(config.test_model_dir, str(config.global_step) + '.pt')
    # 모델 파라미터 로드
    state = torch.load(model_file, weights_oNL_realy=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    # model.load_state_dict(state['model'])
    print('Successfully loaded The model saved at global step = {}'.format(state['global_step']))
    # 모델 평가
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print('using gpu...')
    # 테스트 데이터셋 읽기
    filenames = os.listdir(config.test_data_root)
    for file in filenames:
        if os.path.splitext(file)[1] == '.mat':
            # 테스트 데이터의 절대 경로
            name = config.test_data_root + '/' + file
            # 테스트 데이터 로드
            tmp = sio.loadmat(name)
            inpts, inptr = np.array(tmp[config.test_varible[0]]), np.array(tmp[config.test_varible[1]])
            inpts, inptr = inpts.T, inptr.T
            # s-simulated(시뮬레이션 데이터), r-real(실제 데이터)
            nums, spec = inpts.shape
            numr, _ = inptr.shape
            # DCT 변환
            for idx in range(nums):
                inpts[idx, :] = dct(np.squeeze(inpts[idx, :]), norm='ortho')
            for idx in range(numr):
                inptr[idx, :] = dct(np.squeeze(inptr[idx, :]), norm='ortho')
            # 3차원 텐서로 변환
            inpts, inptr = np.array([inpts]).reshape((nums, 1, spec)), np.array([inptr]).reshape((numr, 1, spec))
            inpts, inptr = torch.from_numpy(inpts).float(), torch.from_numpy(inptr).float()
            # * inpts, inptr .float() 추가

            # inptt-total
            inptt = torch.cat([inpts, inptr], dim=0)
            # 작은 배치로 나누어 배치 테스트 수행
            test_size = 32
            group_total = torch.split(inptt, test_size)
            # 테스트 결과 저장
            predt = []
            # preds, predr = [], []
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
            # 테스트 결과 저장 디렉토리 위치 가져오기
            test_dir = test_result_dir(config)

            # 새로운 절대 경로 파일 이름
            # filename = os.path.join(test_dir, "".join(file))
            # 테스트 데이터셋 이름에 '_result' 추가
            base_name, ext = os.path.splitext(file)  # 파일 이름과 확장자 분리
            filename = os.path.join(test_dir, f"{base_name}_result{ext}")  # '_result'를 이름 뒤에 추가

            # 테스트 결과를 테스트 폴더에 저장
            sio.savemat(filename, tmp)


def predict(config):
    print('predicting...')
    # 모델 생성
    model = eval("{}(1,1)".format(config.model_name))
    # 저장된 모델 경로 가져오기
    save_model_path = save_model_dir(config)
    model_file = os.path.join(save_model_path, str(config.global_step) + '.pt')
    # 모델 파라미터 로드
    state = torch.load(model_file, weights_oNL_realy=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    print('Successfully loaded The model saved at global step = {}'.format(state['global_step']))
    # 모델 평가
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print('using gpu...')

    # 필요한 interpolation 모듈 임포트
    from scipy.interpolate import interp1d

    # 처리가 필요한 txt 파일 읽기
    filenames = os.listdir(config.predict_root)
    plt.figure(figsize=(9,7))

    for file in filenames:
        if os.path.splitext(file)[1] == '.txt':
            # 테스트 데이터의 절대 경로
            name = os.path.join(config.predict_root, file)
            new_name = os.path.join(config.predict_save_root, 'dn_' + file)
            # 테스트 데이터 로드 (가정: 첫 번째 열은 wave, 두 번째 열은 intensity)
            tmp = np.loadtxt(name, delimiter='\t').astype(float)
            wave, x = tmp[:, 0], tmp[:, 1]

            # ---- 보간(interpolation) 추가: 스펙트럼 길이를 1600 포인트로 확장 ----
            row_new_length = 1600
            # 보간할 새로운 wave 축 생성 (원래 범위 내에서)
            wave_new = np.linspace(wave.min(), wave.max(), row_new_length)
            # cubic 보간 함수 생성
            f_interp = interp1d(wave, x, kind='cubic')
            # 보간된 intensity 값
            x_interp = f_interp(wave_new)
            # ---------------------------------------------------------------------

            # DCT 변환: 보간된 스펙트럼 사용
            coe_dct = dct(x_interp, norm='ortho')
            # 형태 변경: [batch, channel, spec_length]
            inpt = coe_dct.reshape(1, 1, -1)
            # torch 텐서로 변환
            inpt = torch.from_numpy(inpt).float()
            if torch.cuda.is_available():
                inpt = inpt.cuda()

            # 모델 예측
            yt = model(inpt).detach().cpu().numpy()
            yt = yt.reshape(-1, )
            # inverse DCT 변환: 예측된 노이즈를 원래 도메인으로 변환
            noise = idct(yt, norm='ortho')
            # denoised 스펙트럼: 보간된 스펙트럼에서 예측된 노이즈를 빼줌
            Y = x_interp - noise

            # 결과 저장: wave_new와 denoised 스펙트럼을 저장
            denoised = np.array([wave_new, Y])
            np.savetxt(new_name, denoised, delimiter='\t')
            
            # 시각화: 원본 보간 스펙트럼과 denoised 스펙트럼 플롯
            plt.plot(wave_new, x_interp, label='Noisy Spectrum')
            plt.plot(wave_new, Y, label='Denoised Spectrum')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()


def batch_predict(config):
    print('batch predicting...')
    # 모델 생성
    model = eval("{}(1,1)".format(config.model_name))
    # 저장된 모델 경로 가져오기
    # save_model_path = save_model_dir(config)
    model_file = os.path.join(config.test_model_dir, str(config.global_step) + '.pt')
    # 모델 파라미터 로드
    # state = torch.load(model_file)
    state = torch.load(model_file, weights_oNL_realy=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    # model.load_state_dict(state['model'])
    print('Successfully loaded The model saved at global step = {}'.format(state['global_step']))
    # 모델 평가
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print('using gpu...')
    # 테스트 데이터셋 읽기
    filenames = os.listdir(config.predict_root)
    for file in filenames:
        if os.path.splitext(file)[1] == '.mat':
            # 테스트 데이터의 절대 경로
            name = config.predict_root + '/' + file
            # 테스트 데이터 로드
            tmp = sio.loadmat(name)
            inpts = np.array(tmp['cube'])
            inpts = np.array(tmp['spectrum'])
            print(f"Shape of spectrum: {inpts.shape}")
            print(f"Data example: {inpts[:5]}")  # 일부 데이터 출력

            inpts = inpts.T
            nums, spec = inpts.shape
            # DCT 변환
            for idx in range(nums):
                inpts[idx, :] = dct(np.squeeze(inpts[idx, :]), norm='ortho')
            # 3차원 텐서로 변환
            inpts = np.array([inpts]).reshape((nums, 1, spec))
            inpts = torch.from_numpy(inpts).float()

            # 작은 배치로 나누어 배치 테스트 수행
            test_size = 32
            group_total = torch.split(inpts, test_size)
            # 테스트 결과 저장
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
            # 테스트 결과 저장 디렉토리 위치 가져오기
            test_dir = test_result_dir(config)
            # 새로운 절대 경로 파일 이름
            filename = os.path.join(test_dir, "".join(file))
            # 테스트 결과를 테스트 폴더에 저장
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
