import numpy as np
from collections import OrderedDict
from common.layers import Convolution, Relu, Pooling, Affine, Dropout
# TimeSoftmaxWithLoss는 별도의 파일로 관리하거나 layers.py에 합칠 수 있습니다.
# 여기서는 common.layers에 포함되어 있다고 가정합니다.
from common.layers import TimeSoftmaxWithLoss

class SimpleConvNet:
    """
    모든 Convolution 계층의 Stride가 2로 설정된 버전.
    - 구조: Conv(s=2)-ReLU-Conv(s=2)-ReLU-Pool(s=2) ...
    - Stride 변경으로 인한 차원 문제를 해결하기 위해 Pool2 계층이 제거되었습니다.
    """
    def __init__(self, input_dim=(1, 64, 256),
                 conv_params=[
                     # ✨✨✨ stride를 모두 2로 변경 ✨✨✨
                     {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 2},
                     {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 2},
                     {'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 2},
                     {'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 2}],
                 hidden_size=128,
                 max_label_len=25,
                 vocab_size=4109):
        
        self.max_label_len = max_label_len
        self.vocab_size = vocab_size

        # 1. 가중치 초기화
        self.params = {}
        pre_channel_num = input_dim[0]
        
        for i, param in enumerate(conv_params):
            scale = np.sqrt(2.0 / (pre_channel_num * param['filter_size'] * param['filter_size']))
            self.params[f'W{i+1}'] = scale * np.random.randn(param['filter_num'], pre_channel_num, param['filter_size'], param['filter_size'])
            self.params[f'b{i+1}'] = np.zeros(param['filter_num'])
            pre_channel_num = param['filter_num']
        
        # ✨✨✨ conv_output_size 재계산 ✨✨✨
        # 입력 (H,W): (64, 256)
        # Conv1(s=2): (32, 128)
        # Conv2(s=2): (16, 64)
        # Pool1(s=2): (8, 32)
        # Conv3(s=2): (4, 16)
        # Conv4(s=2): (2, 8)
        # 최종 특징 맵 크기: (32 channels, 2 height, 8 width)
        conv_output_size = 32 * 2 * 8  # 128 -> 512
        
        scale = np.sqrt(2.0 / conv_output_size)
        self.params['W5'] = scale * np.random.randn(conv_output_size, hidden_size)
        self.params['b5'] = np.zeros(hidden_size)
        
        scale = np.sqrt(2.0 / hidden_size)
        self.params['W6'] = scale * np.random.randn(hidden_size, max_label_len * vocab_size)
        self.params['b6'] = np.zeros(max_label_len * vocab_size)

        # 2. 계층 생성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_params[0]['stride'], conv_params[0]['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], conv_params[1]['stride'], conv_params[1]['pad'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'], conv_params[2]['stride'], conv_params[2]['pad'])
        self.layers['Relu3'] = Relu()
        self.layers['Conv4'] = Convolution(self.params['W4'], self.params['b4'], conv_params[3]['stride'], conv_params[3]['pad'])
        self.layers['Relu4'] = Relu()
        # ✨✨✨ Pool2 계층 제거 ✨✨✨
        # self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Affine1'] = Affine(self.params['W5'], self.params['b5'])
        self.layers['Relu5'] = Relu()
        self.layers['Dropout'] = Dropout(0.05)
        
        self.layers['Affine2'] = Affine(self.params['W6'], self.params['b6'])

        self.last_layer = TimeSoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=True):
        y = self.predict(x, train_flg)
        y = y.reshape(-1, self.max_label_len, self.vocab_size)
        loss = self.last_layer.forward(y, t)
        return loss

    def accuracy(self, x, t, pad_id=0):
        y = self.predict(x, train_flg=False)
        y = y.reshape(-1, self.max_label_len, self.vocab_size)
        y = np.argmax(y, axis=2)

        correct_words = 0
        for i in range(len(t)):
            pred_word = y[i]
            true_word = t[i]
            
            true_word_end_idx = np.where(true_word == pad_id)[0]
            true_len = true_word_end_idx[0] if len(true_word_end_idx) > 0 else self.max_label_len
            
            if np.array_equal(pred_word[:true_len], true_word[:true_len]):
                correct_words += 1
        
        return correct_words / len(t)

    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)

        dout = self.last_layer.backward(1)
        dout = dout.reshape(-1, self.max_label_len * self.vocab_size)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # ✨✨✨ 그래디언트 저장 버그 수정 ✨✨✨
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Conv3'].dW, self.layers['Conv3'].db
        grads['W4'], grads['b4'] = self.layers['Conv4'].dW, self.layers['Conv4'].db
        grads['W5'], grads['b5'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W6'], grads['b6'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        
        return grads