import cupy as cp
import numpy as np
from collections import OrderedDict
from common.cu_layers import Convolution, Relu, Pooling, Affine, Dropout, BatchNormalization, TimeSoftmaxWithLoss

class OcrNetBN:
    """
    배치 정규화(BN)가 적용된 OCR 모델. TimeSoftmaxWithLoss 사용 버전.
    ✨ hidden_size를 256으로 변경.
    """
    def __init__(self, input_dim=(1, 64, 256),
                 conv_params=[
                     {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 2},
                     {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 2},
                     {'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 2},
                     {'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 2}],
                 hidden_size=256, # ✨ 기본값을 128에서 256으로 변경
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

        conv_output_size = 32 * 2 * 8

        # Affine 1 가중치 (✨ 출력 크기가 hidden_size로 변경됨)
        scale = cp.sqrt(2.0 / conv_output_size)
        self.params['W5'] = scale * cp.random.randn(conv_output_size, hidden_size, dtype=cp.float32)
        self.params['b5'] = cp.zeros(hidden_size, dtype=cp.float32) # ✨ 크기 변경
        self.params['gamma5'] = cp.ones(hidden_size, dtype=cp.float32) # ✨ 크기 변경
        self.params['beta5'] = cp.zeros(hidden_size, dtype=cp.float32)  # ✨ 크기 변경

        # Affine 2 가중치 (✨ 입력 크기가 hidden_size로 변경됨)
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
        self.layers['Dropout'] = Dropout(0.05)
        self.layers['Affine2'] = Affine(self.params['W6'], self.params['b6'])

        self.last_layer = TimeSoftmaxWithLoss()

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