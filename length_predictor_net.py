import numpy as np
from collections import OrderedDict
from common.layers import Convolution, Relu, Pooling, Affine, Dropout, SoftmaxWithLoss

class LengthPredictorNet:
    """
    이미지를 입력받아 글자의 '개수'를 예측하는 분류 모델.
    - 출력: 각 길이에 대한 확률 (예: 1글자일 확률, 2글자일 확률 ...)
    - 최종 손실 함수로 SoftmaxWithLoss를 사용합니다.
    """
    def __init__(self, input_dim=(1, 64, 256),
                 conv_params=[
                     {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 2},
                     {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 2},
                     {'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 2},
                     {'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 2}],
                 hidden_size=128,
                 max_output_len=25): # 예측할 최대 글자 수 (클래스 개수)

        self.max_output_len = max_output_len

        # 1. 가중치 초기화
        self.params = {}
        pre_channel_num = input_dim[0]
        
        for i, param in enumerate(conv_params):
            scale = np.sqrt(2.0 / (pre_channel_num * param['filter_size'] * param['filter_size']))
            self.params[f'W{i+1}'] = scale * np.random.randn(param['filter_num'], pre_channel_num, param['filter_size'], param['filter_size'])
            self.params[f'b{i+1}'] = np.zeros(param['filter_num'])
            pre_channel_num = param['filter_num']
        
        # 입력 (64, 256) 기준 CNN 통과 후 크기 계산
        # (64, 256) -> C1(s=2) -> (32, 128) -> C2(s=2) -> (16, 64) -> P1(s=2) -> (8, 32)
        # -> C3(s=2) -> (4, 16) -> C4(s=2) -> (2, 8)
        # 최종 특징 맵: (32 channels, 2 height, 8 width)
        conv_output_size = 32 * 2 * 8
        
        # Affine 계층 가중치
        scale = np.sqrt(2.0 / conv_output_size)
        self.params['W5'] = scale * np.random.randn(conv_output_size, hidden_size)
        self.params['b5'] = np.zeros(hidden_size)
        
        scale = np.sqrt(2.0 / hidden_size)
        # 최종 출력층: 클래스 개수(max_output_len) 만큼의 뉴런을 가짐
        self.params['W6'] = scale * np.random.randn(hidden_size, max_output_len)
        self.params['b6'] = np.zeros(max_output_len)

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
        self.layers['Affine1'] = Affine(self.params['W5'], self.params['b5'])
        self.layers['Relu5'] = Relu()
        self.layers['Dropout'] = Dropout(0.15)
        self.layers['Affine2'] = Affine(self.params['W6'], self.params['b6'])

        # 최종 계층은 SoftmaxWithLoss
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=True):
        """손실 함수
        x: 입력 이미지 데이터
        t: 정답 레이블 (글자의 '길이')
        """
        y = self.predict(x, train_flg)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        """정확도 계산
        t: 정답 레이블 (글자의 '길이')
        """
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        # t가 원-핫 인코딩 형태일 수 있으므로, argmax로 변환
        if t.ndim != 1: t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # 1. 순전파
        self.loss(x, t, train_flg=True)

        # 2. 역전파
        dout = self.last_layer.backward(1)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 3. 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Conv3'].dW, self.layers['Conv3'].db
        grads['W4'], grads['b4'] = self.layers['Conv4'].dW, self.layers['Conv4'].db
        grads['W5'], grads['b5'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W6'], grads['b6'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        
        return grads
