import cupy as cp  # ✨ NumPy를 CuPy로 변경
from collections import OrderedDict
from common.cu_layers import Convolution, Relu, Pooling, Affine, Dropout, SoftmaxWithLoss

class LengthPredictorNet:
    """
    이미지를 입력받아 글자의 '개수'를 예측하는 분류 모델 (CuPy 버전).
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
                 max_output_len=25):

        self.max_output_len = max_output_len

        # 1. 가중치 초기화 (✨ CuPy를 사용하여 GPU에 직접 생성)
        self.params = {}
        pre_channel_num = input_dim[0]
        
        for i, param in enumerate(conv_params):
            # He 초기값 사용
            scale = cp.sqrt(2.0 / (pre_channel_num * param['filter_size'] * param['filter_size']))
            self.params[f'W{i+1}'] = scale * cp.random.randn(param['filter_num'], pre_channel_num, param['filter_size'], param['filter_size'], dtype=cp.float32)
            self.params[f'b{i+1}'] = cp.zeros(param['filter_num'], dtype=cp.float32)
            pre_channel_num = param['filter_num']
        
        conv_output_size = 32 * 2 * 8
        
        # Affine 계층 가중치
        scale = cp.sqrt(2.0 / conv_output_size)
        self.params['W5'] = scale * cp.random.randn(conv_output_size, hidden_size, dtype=cp.float32)
        self.params['b5'] = cp.zeros(hidden_size, dtype=cp.float32)
        
        scale = cp.sqrt(2.0 / hidden_size)
        self.params['W6'] = scale * cp.random.randn(hidden_size, max_output_len, dtype=cp.float32)
        self.params['b6'] = cp.zeros(max_output_len, dtype=cp.float32)

        # 2. 계층 생성 (파라미터들이 이미 CuPy 배열이므로 계층들도 GPU에서 동작)
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
        self.layers['Dropout'] = Dropout(0.05)
        self.layers['Affine2'] = Affine(self.params['W6'], self.params['b6'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=True):
        y = self.predict(x, train_flg)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        """정확도 계산 (✨ CuPy를 사용하여 GPU에서 직접 계산)"""
        y = self.predict(x, train_flg=False)
        y = cp.argmax(y, axis=1)
        if t.ndim != 1: 
            t = cp.argmax(t, axis=1)
        
        accuracy = cp.sum(y == t) / float(x.shape[0])
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

        # 3. 결과 저장 (dW, db 등은 각 계층에서 이미 CuPy 배열로 계산됨)
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Conv3'].dW, self.layers['Conv3'].db
        grads['W4'], grads['b4'] = self.layers['Conv4'].dW, self.layers['Conv4'].db
        grads['W5'], grads['b5'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W6'], grads['b6'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        
        return grads