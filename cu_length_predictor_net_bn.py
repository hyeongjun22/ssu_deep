import cupy as cp
from collections import OrderedDict
# ✨ BatchNormalization을 cu_layers에서 함께 임포트합니다.
from common.cu_layers import Convolution, Relu, Pooling, Affine, Dropout, BatchNormalization, SoftmaxWithLoss

class LengthPredictorNetBN:
    """
    LengthPredictorNet에 배치 정규화(Batch Normalization)를 적용한 버전.
    Conv/Affine 계층과 ReLU 활성화 함수 사이에 BN 계층을 추가합니다.
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

        # 1. 가중치 초기화
        self.params = {}
        pre_channel_num = input_dim[0]
        
        for i, param in enumerate(conv_params):
            # Conv 가중치
            scale = cp.sqrt(2.0 / (pre_channel_num * param['filter_size'] * param['filter_size']))
            self.params[f'W{i+1}'] = scale * cp.random.randn(param['filter_num'], pre_channel_num, param['filter_size'], param['filter_size'], dtype=cp.float32)
            self.params[f'b{i+1}'] = cp.zeros(param['filter_num'], dtype=cp.float32)
            
            # ✨ BN 파라미터 (gamma, beta)
            self.params[f'gamma{i+1}'] = cp.ones(param['filter_num'], dtype=cp.float32)
            self.params[f'beta{i+1}'] = cp.zeros(param['filter_num'], dtype=cp.float32)
            
            pre_channel_num = param['filter_num']
        
        conv_output_size = 32 * 2 * 8
        
        # Affine 1 가중치
        scale = cp.sqrt(2.0 / conv_output_size)
        self.params['W5'] = scale * cp.random.randn(conv_output_size, hidden_size, dtype=cp.float32)
        self.params['b5'] = cp.zeros(hidden_size, dtype=cp.float32)
        # ✨ BN 5 파라미터
        self.params['gamma5'] = cp.ones(hidden_size, dtype=cp.float32)
        self.params['beta5'] = cp.zeros(hidden_size, dtype=cp.float32)
        
        # Affine 2 가중치
        scale = cp.sqrt(2.0 / hidden_size)
        self.params['W6'] = scale * cp.random.randn(hidden_size, max_output_len, dtype=cp.float32)
        self.params['b6'] = cp.zeros(max_output_len, dtype=cp.float32)

        # 2. 계층 생성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_params[0]['stride'], conv_params[0]['pad'])
        self.layers['BN1'] = BatchNormalization(self.params['gamma1'], self.params['beta1']) # ✨ 추가
        self.layers['Relu1'] = Relu()
        
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], conv_params[1]['stride'], conv_params[1]['pad'])
        self.layers['BN2'] = BatchNormalization(self.params['gamma2'], self.params['beta2']) # ✨ 추가
        self.layers['Relu2'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        
        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'], conv_params[2]['stride'], conv_params[2]['pad'])
        self.layers['BN3'] = BatchNormalization(self.params['gamma3'], self.params['beta3']) # ✨ 추가
        self.layers['Relu3'] = Relu()

        self.layers['Conv4'] = Convolution(self.params['W4'], self.params['b4'], conv_params[3]['stride'], conv_params[3]['pad'])
        self.layers['BN4'] = BatchNormalization(self.params['gamma4'], self.params['beta4']) # ✨ 추가
        self.layers['Relu4'] = Relu()
        
        self.layers['Affine1'] = Affine(self.params['W5'], self.params['b5'])
        self.layers['BN5'] = BatchNormalization(self.params['gamma5'], self.params['beta5']) # ✨ 추가
        self.layers['Relu5'] = Relu()
        self.layers['Dropout'] = Dropout(0.15) # BN을 사용하면 드롭아웃 의존도를 낮출 수 있지만, 함께 사용하면 성능이 더 좋아지는 경우도 많습니다.
        
        self.layers['Affine2'] = Affine(self.params['W6'], self.params['b6'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            # Dropout과 BN 계층은 train_flg에 따라 동작이 달라집니다.
            if "Dropout" in key or "BN" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=True):
        y = self.predict(x, train_flg)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = cp.argmax(y, axis=1)
        if t.ndim != 1: 
            t = cp.argmax(t, axis=1)
        
        accuracy = cp.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)

        dout = self.last_layer.backward(1)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # ✨ BN 계층의 그래디언트(dgamma, dbeta)도 추가로 저장합니다.
        grads = {}
        for i in range(1, 6):
            if i < 5: # Conv + BN
                grads[f'W{i}'], grads[f'b{i}'] = self.layers[f'Conv{i}'].dW, self.layers[f'Conv{i}'].db
                grads[f'gamma{i}'], grads[f'beta{i}'] = self.layers[f'BN{i}'].dgamma, self.layers[f'BN{i}'].dbeta
            elif i == 5: # Affine + BN
                grads[f'W{i}'], grads[f'b{i}'] = self.layers[f'Affine1'].dW, self.layers[f'Affine1'].db
                grads[f'gamma{i}'], grads[f'beta{i}'] = self.layers[f'BN{i}'].dgamma, self.layers[f'BN{i}'].dbeta

        # 마지막 Affine 계층
        grads['W6'], grads['b6'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        
        return grads
