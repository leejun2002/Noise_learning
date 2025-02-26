import warnings
import torch


class DefaultConfig(object):
    env = 'default'  # visdom 환경
    model_name = 'UNet_CA6'
    # 모델 저장 디렉토리 정의
    checkpoint = './Checkpoints'
    # 로그 저장 디렉토리 정의
    logs = './Logs'

    # [ train 로그 결과 그래프 ]
    # tensorboard --logdir=Logs --port=6006
    # -> 위 명령어를 /home/username/NL_real 에서 실행

    # 테스트 결과 저장 경로 정의
    test_dir = './Result'
    # 데이터 출처 정의
    # Instrument = 'Hariba'
    Instrument = 'IWSDG_ALL'
    # 테스트할 모델 경로 (필요시 수정)
    # Hariba
    # test_model_dir = r'H:\Projects\Instrumental noise modeling\code\VA4\Checkpoints\Hariba\UNet_CA\batch_64'
    # Nanophoton

    # test_model_dir = r'H:\Projects\Instrumental noise modeling\code\VA4\Checkpoints\Nanophoton\UNet_CA\batch_64'
    test_model_dir = r'/home/jyounglee/NL_real/Checkpoints/IWSDG_ALL/UNet_CA6/batch_64'

    # 학습 데이터 저장 경로
    # train_data_root = r'H:\PAPER\paper writing\Noise learning\Simulate datasets'
    train_data_root = r'/home/jyounglee/NL_real/data/train'
    noise_mat_path = r'/home/jyounglee/NL_real/data/train/noise_data.mat'
    raw_noise_base = r'/home/jyounglee/NL_real/data/given_data/noise_3siwafer'

    # 테스트 데이터 저장 경로
    # test_data_root = r'H:\PAPER\paper writing\Noise learning'
    test_data_root = r'/home/jyounglee/NL_real/data/test'

    # 예측 데이터 저장 경로
    # predict_root = r'H:\PAPER\paper writing\Noise learning\데이터\TERS\20220721 PIC'
    # predict_root = r'H:\PAPER\paper writing\Noise learning\수정Revision\광학공간분해능\gaoyun\nanophoton\test'
    predict_root = r'/home/jyounglee/NL_real/data/predict'
    predict_save_root = r'/home/jyounglee/NL_real/data/predict_save'

    # batch_predict 데이터 저장 경로
    raw_predict_base = r'/home/jyounglee/NL_real/data/given_data/3layer_on_siwafer'
    batch_predict_root = r'/home/jyounglee/NL_real/data/batch_predict'
    batch_save_root = r'/home/jyounglee/NL_real/Result'
    rank_bg = 1

    batch_size = 64  # 배치 크기
    print_freq = 50 # N 배치마다 정보 출력
    max_epoch = 2000
    lr = 0.001  # 초기 학습률
    lr_decay = 0.5  # 검증 손실 증가 시, lr = lr * lr_decay
    fade_factor = 0  # 0 : 배경 완전 제거 / 1 : 배경 제거 X

    wavelet_model = 'coif4'
    wavelet_level = 6
    wavelet_threshold = None
    block_height = 50
    block_width = 50
    num_sv = 1

    use_svd = True
    visualize = True

    # train
    is_training = True
    is_pretrain = False
    is_testing = False
    is_predicting = False
    is_batch_predicting = False

    # predict
    # is_training = False
    # is_pretrain = True
    # is_testing = False
    # is_predicting = True
    # is_batch_predicting = False

    # batch_predict
    # is_training = False
    # is_pretrain = True
    # is_testing = False
    # is_predicting = False
    # is_batch_predicting = True

    # GPU 사용 여부
    use_gpu = True
    # 특정 단계의 모델 로드
    # Hariba
    # Nanophoton
    # global_step = 850000
    global_step = 0
    # 검증 데이터셋 비율 정의
    valid_ratio = 20
    # 입력 데이터 이름 정의
    test_varible = ['lcube', 'cube']

    def _parse(self, kwargs, opt=None):
        """
        딕셔너리 kwargs를 기반으로 config 매개변수 업데이트
        """
        for k, v in kwargs.items():  # 딕셔너리 items 메서드는 (키, 값) 튜플 배열 반환
            if not hasattr(self, k):  # k 속성이 없으면 경고 출력
                warnings.warn("경고: opt에 %s 속성이 없습니다." % k)
            setattr(self, k, v)  # k 속성에 값 설정

        opt.device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')
        '''
        torch.device는 torch.Tensor를 할당할 장치 객체를 나타냅니다.
        torch.device는 장치 유형('cpu' 또는 'cuda')과 선택적 장치 번호를 포함합니다.
        장치 번호가 없으면 현재 장치를 사용합니다. 예를 들어,
        'cuda' 장치로 생성된 torch.Tensor는 'cuda:X'와 같습니다.
        여기서 X는 torch.cuda.current_device()의 결과입니다.
        torch.Tensor의 장치는 Tensor.device 속성을 통해 접근할 수 있습니다.
        torch.device는 문자열/문자열 및 장치 번호를 통해 생성됩니다.
        '''

        print('사용자 구성:')
        for k, v in self.__class__.__dict__.items():  # 인스턴스의 클래스 속성
            if not k.startswith('__'):  # '__'로 시작하지 않는 경우
                print(k, getattr(self, k))  # 값 출력