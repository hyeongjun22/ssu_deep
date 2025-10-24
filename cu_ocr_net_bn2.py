import cupy as cp
import numpy as np
from collections import OrderedDict
from common.cu_layers import Convolution, Relu, Pooling, Affine, Dropout, BatchNormalization, TimeSoftmaxWithLoss, TimeSoftmaxWithLossMasked

class OcrNetBN:
    """
    배치 정규화(BN)가 적용된 OCR 모델. TimeSoftmaxWithLoss 사용 버전.
    ✨ 모델 용량 증가: CNN 필터 수 및 Hidden Size 상향 조정.
    """
    def __init__(self, input_dim=(1, 64, 256),
                 conv_params=[
                     # ✨ 변경: 16 -> 32
                     {'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 2},
                     # ✨ 변경: 16 -> 64
                     {'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 2},
                     # ✨ 변경: 32 -> 128
                     {'filter_num': 128, 'filter_size': 3, 'pad': 1, 'stride': 2},
                     # ✨ 변경: 32 -> 128
                     {'filter_num': 128, 'filter_size': 3, 'pad': 1, 'stride': 2}],
                 hidden_size=512, # ✨ 변경: 256 -> 512
                 max_label_len=25,
                 vocab_size=4109):

        self.max_label_len = max_label_len
        self.vocab_size = vocab_size

        # 1. 가중치 초기화
        self.params = {}
        pre_channel_num = input_dim[0]

        for i, param in enumerate(conv_params):
            scale = cp.sqrt(2.0 / (pre_channel_num * param['filter_size'] * param['filter_size']))
            self.params[f'W{i+1}'] = scale * cp.random.randn(param['filter_num'], pre_channel_num, param['filter_size'], param['filter_size'], dtype=cp.float32)
            self.params[f'b{i+1}'] = cp.zeros(param['filter_num'], dtype=cp.float32)
            self.params[f'gamma{i+1}'] = cp.ones(param['filter_num'], dtype=cp.float32)
            self.params[f'beta{i+1}'] = cp.zeros(param['filter_num'], dtype=cp.float32)
            pre_channel_num = param['filter_num']

        # ✨ 중요: CNN 출력 크기를 동적으로 계산
        # (Input H, W) = (64, 256)
        # (Conv1, S2) -> (32, 128)
        # (Conv2, S2) -> (16, 64)
        # (Pool1, S2) -> (8, 32)
        # (Conv3, S2) -> (4, 16)
        # (Conv4, S2) -> (2, 8)
        # CNN 최종 출력 형상: (N, 128, 2, 8)
        conv_output_h = 2
        conv_output_w = 8
        # ✨ conv_params의 마지막 필터 수를 가져오도록 변경 (128)
        last_filter_num = conv_params[-1]['filter_num']
        conv_output_size = last_filter_num * conv_output_h * conv_output_w

        # Affine 1 가중치 (입력: conv_output_size, 출력: hidden_size)
        scale = cp.sqrt(2.0 / conv_output_size)
        self.params['W5'] = scale * cp.random.randn(conv_output_size, hidden_size, dtype=cp.float32)
        self.params['b5'] = cp.zeros(hidden_size, dtype=cp.float32)
        self.params['gamma5'] = cp.ones(hidden_size, dtype=cp.float32)
        self.params['beta5'] = cp.zeros(hidden_size, dtype=cp.float32)

        # Affine 2 가중치 (입력: hidden_size, 출력: max_label_len * vocab_size)
        scale = cp.sqrt(2.0 / hidden_size)
        output_size = max_label_len * vocab_size
        self.params['W6'] = scale * cp.random.randn(hidden_size, output_size, dtype=cp.float32)
        self.params['b6'] = cp.zeros(output_size, dtype=cp.float32)

        # 2. 계층 생성 (순서는 동일)
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_params[0]['stride'], conv_params[0]['pad'])
        self.layers['BN1'] = BatchNormalization(self.params['gamma1'], self.params['beta1'])
        self.layers['Relu1'] = Relu()
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], conv_params[1]['stride'], conv_params[1]['pad'])
        self.layers['BN2'] = BatchNormalization(self.params['gamma2'], self.params['beta2'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'], conv_params[2]['stride'], conv_params[2]['pad'])
        self.layers['BN3'] = BatchNormalization(self.params['gamma3'], self.params['beta3'])
        self.layers['Relu3'] = Relu()
        self.layers['Conv4'] = Convolution(self.params['W4'], self.params['b4'], conv_params[3]['stride'], conv_params[3]['pad'])
        self.layers['BN4'] = BatchNormalization(self.params['gamma4'], self.params['beta4'])
        self.layers['Relu4'] = Relu()
        self.layers['Affine1'] = Affine(self.params['W5'], self.params['b5'])
        self.layers['BN5'] = BatchNormalization(self.params['gamma5'], self.params['beta5'])
        self.layers['Relu5'] = Relu()
        self.layers['Dropout'] = Dropout(0.05) # (필요시 이 값도 0.3 등으로 늘릴 수 있습니다)
        self.layers['Affine2'] = Affine(self.params['W6'], self.params['b6'])

        self.last_layer = TimeSoftmaxWithLossMasked(pad_id=0)

    def predict(self, x, train_flg=False):
        # (이하 predict, loss, accuracy, gradient 메서드는 변경 없음)
        for key, layer in self.layers.items():
            if "Dropout" in key or "BN" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=True):
        y = self.predict(x, train_flg)
        N = y.shape[0]
        y_reshaped = y.reshape(N, self.max_label_len, self.vocab_size)
        return self.last_layer.forward(y_reshaped, t)

    def accuracy(self, x, t, pad_id=0):
        y = self.predict(x, train_flg=False)
        N = y.shape[0]
        y_reshaped = y.reshape(N, self.max_label_len, self.vocab_size)
        y_pred = cp.argmax(y_reshaped, axis=2)

        correct_words = 0
        y_pred_np = cp.asnumpy(y_pred)
        t_np = cp.asnumpy(t)

        for i in range(N):
            pred_word = y_pred_np[i]
            true_word = t_np[i]
            true_word_end_idx = np.where(true_word == pad_id)[0]
            true_len = true_word_end_idx[0] if len(true_word_end_idx) > 0 else self.max_label_len
            if np.array_equal(pred_word[:true_len], true_word[:true_len]):
                correct_words += 1
        return correct_words / N

    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)
        dout = self.last_layer.backward(1)
        N = dout.shape[0]
        dout = dout.reshape(N, -1)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        for i in range(1, 6):
            if i < 5:
                grads[f'W{i}'], grads[f'b{i}'] = self.layers[f'Conv{i}'].dW, self.layers[f'Conv{i}'].db
                grads[f'gamma{i}'], grads[f'beta{i}'] = self.layers[f'BN{i}'].dgamma, self.layers[f'BN{i}'].dbeta
            elif i == 5:
                grads[f'W{i}'], grads[f'b{i}'] = self.layers[f'Affine1'].dW, self.layers[f'Affine1'].db
                grads[f'gamma{i}'], grads[f'beta{i}'] = self.layers[f'BN{i}'].dgamma, self.layers[f'BN{i}'].dbeta
        grads['W6'], grads['b6'] = self.layers[f'Affine2'].dW, self.layers[f'Affine2'].db
        return grads