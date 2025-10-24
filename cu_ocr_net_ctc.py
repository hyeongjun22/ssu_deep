import cupy as cp
import numpy as np
from collections import OrderedDict
# ✨ common.cu_layers에서 CTCLoss, TimeLSTM, TimeAffine 등 필요한 모든 것을 임포트
from common.cu_layers import (
    Convolution, Relu, Pooling, Affine, Dropout, BatchNormalization, 
    TimeLSTM, TimeAffine, CTCLoss
)

class OcrNetCTC:
    """
    CNN + RNN (LSTM) + CTCLoss 구조의 OCR 모델
    """
    def __init__(self, input_dim=(1, 64, 256),
                 conv_params=[
                     {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1}, # Conv1
                     {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1}, # Conv2
                     # Pool1은 아래 self.layers에서 별도 정의됨
                     {'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1}, # Conv3 (stride 2 -> 1 수정)
                     {'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1}, # Conv4
                     # Pool2는 아래 self.layers에서 별도 정의됨
                     {'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1}, # Conv5 (stride 2 -> 1 수정)
                     {'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1}, # Conv6
                 ],
                 rnn_hidden_size=256,
                 vocab_size=4109,
                 blank_id=0): # 0번을 <PAD> 겸 <BLANK>로 사용

        self.vocab_size = vocab_size
        self.blank_id = blank_id
        
        # 1. 가중치 초기화
        self.params = {}
        pre_channel_num = input_dim[0]
        
        # CNN (Feature Extractor)
        # (W, H) = (256, 64)
        # Conv1,2: (256, 64) -> (256, 64)
        # Pool1: (256, 64) -> (128, 32)
        # Conv3,4: (128, 32) -> (128, 32)
        # Pool2: (128, 32) -> (64, 16)
        # Conv5,6: (64, 16) -> (64, 16)
        # 최종 CNN 출력: (N, 64, 16, 64) -> (N, C, H, W)
        
        for i, param in enumerate(conv_params):
            scale = cp.sqrt(2.0 / (pre_channel_num * param['filter_size'] * param['filter_size']))
            self.params[f'W{i+1}'] = scale * cp.random.randn(param['filter_num'], pre_channel_num, param['filter_size'], param['filter_size'], dtype=cp.float32)
            self.params[f'b{i+1}'] = cp.zeros(param['filter_num'], dtype=cp.float32)
            self.params[f'gamma{i+1}'] = cp.ones(param['filter_num'], dtype=cp.float32)
            self.params[f'beta{i+1}'] = cp.zeros(param['filter_num'], dtype=cp.float32)
            pre_channel_num = param['filter_num']

        # CNN 출력 크기 계산
        self.cnn_output_w = 64
        self.cnn_output_h = 16
        self.cnn_output_c = 64
        
        # RNN (Sequence Processor)
        # 입력 차원 D = C * H = 64 * 16 = 1024
        # 시퀀스 길이 T = W = 64
        rnn_input_size = self.cnn_output_c * self.cnn_output_h # 1024
        H = rnn_hidden_size # 256
        
        scale_lstm = cp.sqrt(1.0 / rnn_input_size)
        self.params['LSTM_Wx'] = scale_lstm * cp.random.randn(rnn_input_size, 4 * H, dtype=cp.float32)
        self.params['LSTM_Wh'] = scale_lstm * cp.random.randn(H, 4 * H, dtype=cp.float32)
        self.params['LSTM_b'] = cp.zeros(4 * H, dtype=cp.float32)
        
        # Time Affine (Output Layer)
        # 출력 차원 V = vocab_size (blank 포함)
        # (blank_id=0이므로 vocab_size가 4109라면 0~4108까지 V=4109)
        V = self.vocab_size 
        scale_affine = cp.sqrt(1.0 / H)
        self.params['Affine_W'] = scale_affine * cp.random.randn(H, V, dtype=cp.float32)
        self.params['Affine_b'] = cp.zeros(V, dtype=cp.float32)

        # 2. 계층 생성
        self.layers = OrderedDict()
        
        # CNN Layers
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
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Conv5'] = Convolution(self.params['W5'], self.params['b5'], conv_params[4]['stride'], conv_params[4]['pad'])
        self.layers['BN5'] = BatchNormalization(self.params['gamma5'], self.params['beta5'])
        self.layers['Relu5'] = Relu()
        self.layers['Conv6'] = Convolution(self.params['W6'], self.params['b6'], conv_params[5]['stride'], conv_params[5]['pad'])
        self.layers['BN6'] = BatchNormalization(self.params['gamma6'], self.params['beta6'])
        self.layers['Relu6'] = Relu()
        
        # RNN Layers
        self.layers['LSTM'] = TimeLSTM(self.params['LSTM_Wx'], self.params['LSTM_Wh'], self.params['LSTM_b'])
        self.layers['Affine'] = TimeAffine(self.params['Affine_W'], self.params['Affine_b'])
        
        self.last_layer = CTCLoss(blank_id=self.blank_id)
        
        # BN 계층의 파라미터를 params 딕셔너리에 추가 (Adam 옵티마이저가 인식하도록)
        # (Conv 5, 6 추가)
        for i in range(1, 7):
            self.params[f'gamma{i}'] = self.layers[f'BN{i}'].gamma
            self.params[f'beta{i}'] = self.layers[f'BN{i}'].beta

    def predict(self, x, train_flg=False):
        # 1. CNN
        for key, layer in self.layers.items():
            if "LSTM" in key or "Affine" in key:
                break # RNN 전까지
            if "Dropout" in key or "BN" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        
        # 2. Reshape (Map-to-Sequence)
        # x.shape = (N, C, H, W) = (N, 64, 16, 64)
        N, C, H, W = x.shape
        # (N, W, C, H)
        x = x.transpose(0, 3, 1, 2)
        # (N, W, C * H) = (N, T, D) = (N, 64, 1024)
        x = x.reshape(N, W, -1)
        self.cnn_output_len = cp.array([W] * N, dtype=cp.int32) # (N,)
        
        # 3. RNN
        x = self.layers['LSTM'].forward(x)
        x = self.layers['Affine'].forward(x)
        
        return x # (N, T, V)

    def loss(self, x, t, t_len, train_flg=True):
        """
        x: (N, 1, 64, 256) - 입력 이미지
        t: (N, S_max) - 패딩된 정답 레이블
        t_len: (N,) - 각 레이블의 실제 길이
        """
        y_pred = self.predict(x, train_flg)
        # y_pred: (N, T, V), T=64
        
        # cnn_output_len은 (N,) 크기이며, 모든 요소가 T (64)
        return self.last_layer.forward(y_pred, t, self.cnn_output_len, t_len)

    def gradient(self, x, t, t_len):
        # 1. Forward
        self.loss(x, t, t_len, train_flg=True)
        
        # 2. Backward
        dout = self.last_layer.backward(1) # (N, T, V)
        
        # RNN Backward
        dout = self.layers['Affine'].backward(dout) # (N, T, D_rnn)
        dout = self.layers['LSTM'].backward(dout) # (N, T, D_cnn)
        
        # Reshape (Sequence-to-Map)
        # dout.shape = (N, T, D) = (N, 64, 1024)
        N, W, D = dout.shape
        # (N, W, C, H)
        dout = dout.reshape(N, W, self.cnn_output_c, self.cnn_output_h)
        # (N, C, H, W)
        dout = dout.transpose(0, 2, 3, 1) # (N, 64, 16, 64)
        
        # CNN Backward
        cnn_layers = list(self.layers.items())
        cnn_layers.reverse()
        
        for key, layer in cnn_layers:
            if "LSTM" in key or "Affine" in key:
                continue
            dout = layer.backward(dout)
            
        # 3. 그래디언트 수집
        grads = {}
        # CNN
        for i in range(1, 7): # 6개 Conv
            grads[f'W{i}'], grads[f'b{i}'] = self.layers[f'Conv{i}'].dW, self.layers[f'Conv{i}'].db
            grads[f'gamma{i}'], grads[f'beta{i}'] = self.layers[f'BN{i}'].dgamma, self.layers[f'BN{i}'].dbeta
        
        # RNN
        grads['LSTM_Wx'], grads['LSTM_Wh'], grads['LSTM_b'] = self.layers['LSTM'].grads
        grads['Affine_W'], grads['Affine_b'] = self.layers['Affine'].grads
        
        return grads, self.last_layer.loss

    def greedy_decoder(self, y_pred_softmax, y_len):
        """
        Greedy (Best Path) Decoder
        y_pred_softmax: (N, T, V)
        y_len: (N,) - 각 y_pred의 실제 길이 (T)
        """
        N, T, V = y_pred_softmax.shape
        
        # 1. argmax
        y_argmax = cp.argmax(y_pred_softmax, axis=2) # (N, T)
        
        decoded_words = []
        y_argmax_np = cp.asnumpy(y_argmax)
        y_len_np = cp.asnumpy(y_len)
        
        for i in range(N):
            seq = y_argmax_np[i, :y_len_np[i]]
            
            # 2. Collapse repeats
            collapsed_seq = [seq[0]]
            for j in range(1, len(seq)):
                if seq[j] != seq[j-1]:
                    collapsed_seq.append(seq[j])
            
            # 3. Remove blanks
            decoded_word = [c for c in collapsed_seq if c != self.blank_id]
            decoded_words.append(decoded_word)
            
        return decoded_words

    def accuracy(self, x, t, t_len, pad_id=0):
        """
        t: (N, S_max) - 패딩된 정답 레이블
        t_len: (N,) - 각 레이블의 실제 길이
        """
        y_pred = self.predict(x, train_flg=False) # (N, T, V)
        
        # Softmax
        y_pred_exp = cp.exp(y_pred - cp.max(y_pred, axis=2, keepdims=True))
        y_softmax = y_pred_exp / cp.sum(y_pred_exp, axis=2, keepdims=True)
        
        # 1. Decode prediction
        decoded_words = self.greedy_decoder(y_softmax, self.cnn_output_len)
        
        # 2. Get true words
        t_np = cp.asnumpy(t)
        t_len_np = cp.asnumpy(t_len)
        
        true_words = []
        for i in range(t_np.shape[0]):
            true_word = t_np[i, :t_len_np[i]]
            true_words.append(list(true_word))
            
        # 3. Compare
        correct_words = 0
        for pred, true in zip(decoded_words, true_words):
            if pred == true:
                correct_words += 1
                
        return correct_words / len(true_words)
