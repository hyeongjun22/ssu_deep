import cupy as cp
import numpy as np # 스케일 계산에만 사용
from collections import OrderedDict
from common.cu_layers import Convolution, Relu, Pooling, Affine, Dropout, BatchNormalization
from common.cu_layers import TimeLSTM, TimeAffine, TimeSoftmaxWithLoss

class CuCRNNNet:
    """
    CuPy 기반 CNN + LSTM 구조의 OCR 모델.
    - Backward 로직 수정: 각 계층의 반환값 명시적 처리
    """
    def __init__(self, input_dim=(1, 64, 256),
                 conv_params=[
                     {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                     {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1}, # Pool1
                     {'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                     {'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1}, # Pool2
                     {'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                     {'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1}, # Pool3
                 ],
                 lstm_hidden_size=256,
                 max_label_len=16,
                 vocab_size=4109):

        self.max_label_len = max_label_len
        self.vocab_size = vocab_size
        self.conv_params = conv_params # 나중에 gradient에서 사용하기 위해 저장

        # --- 1. 가중치 초기화 ---
        self.params = {}
        pre_channel_num = input_dim[0]
        cnn_layer_idx = 1
        pool_count = 0

        for i, param in enumerate(conv_params):
            scale = np.sqrt(2.0 / (pre_channel_num * param['filter_size'] * param['filter_size']))
            self.params[f'W{cnn_layer_idx}'] = scale * cp.random.randn(param['filter_num'], pre_channel_num, param['filter_size'], param['filter_size'], dtype=cp.float32)
            self.params[f'b{cnn_layer_idx}'] = cp.zeros(param['filter_num'], dtype=cp.float32)
            self.params[f'gamma{cnn_layer_idx}'] = cp.ones(param['filter_num'], dtype=cp.float32)
            self.params[f'beta{cnn_layer_idx}'] = cp.zeros(param['filter_num'], dtype=cp.float32)
            pre_channel_num = param['filter_num']
            cnn_layer_idx += 1
            if (i + 1) % 2 == 0:
                 pool_count += 1

        cnn_out_c = conv_params[-1]['filter_num']
        cnn_out_h = input_dim[1] // (2**pool_count)
        cnn_out_w = input_dim[2] // (2**pool_count)
        rnn_input_size = cnn_out_c * cnn_out_h

        D, H = rnn_input_size, lstm_hidden_size
        scale_lstm = np.sqrt(1.0 / D)
        self.params['LSTM_Wx'] = scale_lstm * cp.random.randn(D, 4 * H, dtype=cp.float32)
        scale_lstm_h = np.sqrt(1.0 / H)
        self.params['LSTM_Wh'] = scale_lstm_h * cp.random.randn(H, 4 * H, dtype=cp.float32)
        self.params['LSTM_b'] = cp.zeros(4 * H, dtype=cp.float32)

        scale_affine = np.sqrt(1.0 / H)
        self.params['TimeAffine_W'] = scale_affine * cp.random.randn(H, vocab_size, dtype=cp.float32)
        self.params['TimeAffine_b'] = cp.zeros(vocab_size, dtype=cp.float32)

        # --- 2. 계층 생성 ---
        self.cnn_layers = OrderedDict()
        cnn_layer_idx = 1
        pool_idx = 1
        for i, param in enumerate(conv_params):
             self.cnn_layers[f'Conv{cnn_layer_idx}'] = Convolution(self.params[f'W{cnn_layer_idx}'], self.params[f'b{cnn_layer_idx}'], param.get('stride', 1), param.get('pad', 0)) # stride, pad 기본값 처리
             self.cnn_layers[f'BN{cnn_layer_idx}'] = BatchNormalization(self.params[f'gamma{cnn_layer_idx}'], self.params[f'beta{cnn_layer_idx}'])
             self.cnn_layers[f'Relu{cnn_layer_idx}'] = Relu()
             cnn_layer_idx += 1
             if (i + 1) % 2 == 0:
                 self.cnn_layers[f'Pool{pool_idx}'] = Pooling(pool_h=2, pool_w=2, stride=2)
                 pool_idx += 1

        self.rnn_layers = OrderedDict()
        self.rnn_layers['LSTM'] = TimeLSTM(self.params['LSTM_Wx'], self.params['LSTM_Wh'], self.params['LSTM_b'], stateful=False)
        self.rnn_layers['TimeAffine'] = TimeAffine(self.params['TimeAffine_W'], self.params['TimeAffine_b'])

        self.last_layer = TimeSoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        # CNN Forward
        for key, layer in self.cnn_layers.items():
            if "BN" in key or "Dropout" in key:
                 x = layer.forward(x, train_flg)
            else:
                 x = layer.forward(x)
        N, C, H, W = x.shape
        x = x.transpose(0, 3, 1, 2).reshape(N, W, C * H) # RNN 입력 형태로 변환
        # RNN Forward
        for layer in self.rnn_layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=True):
        y = self.predict(x, train_flg)
        N, T, V = y.shape
        if T > self.max_label_len:
            y = y[:, :self.max_label_len, :]
        elif T < self.max_label_len:
            pad_shape = ((0,0), (0, self.max_label_len - T), (0,0))
            y = cp.pad(y, pad_shape, 'constant', constant_values=-cp.inf) # Softmax 입력에 안전한 패딩
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, pad_id=0):
        y = self.predict(x, train_flg=False)
        N, T, V = y.shape
        current_max_len = T # 예측된 시퀀스 길이 사용
        if T > self.max_label_len:
            y = y[:, :self.max_label_len, :]
            current_max_len = self.max_label_len
        elif T < self.max_label_len:
             t = t[:, :T] # 정답 길이를 예측 길이에 맞춤

        y_pred = cp.argmax(y, axis=2)

        correct_words = 0
        N = len(t)
        if N == 0: return 0.0

        y_pred_np = cp.asnumpy(y_pred)
        t_np = cp.asnumpy(t)

        for i in range(N):
            pred_word = y_pred_np[i]
            true_word = t_np[i]
            true_word_end_idx = np.where(true_word == pad_id)[0]
            true_len = true_word_end_idx[0] if len(true_word_end_idx) > 0 else current_max_len
            if true_len > 0 and np.array_equal(pred_word[:true_len], true_word[:true_len]):
                correct_words += 1
        return float(correct_words) / N

    # --- ✨ gradient 메소드 최종 수정 ✨ ---
    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = self.last_layer.backward(1) # (N, T', V)

        # RNN 출력 T와 Loss 출력 T' 차이 처리
        N, T_loss, V = dout.shape
        if self.rnn_layers['LSTM'].hs_shape is None:
             pool_count = len([k for k in self.cnn_layers if k.startswith('Pool')])
             T_rnn = x.shape[3] // (2**pool_count)
        else:
            N_rnn, T_rnn, H_rnn = self.rnn_layers['LSTM'].hs_shape

        if T_loss < T_rnn:
            pad_width = ((0, 0), (0, T_rnn - T_loss), (0, 0))
            dout = cp.pad(dout, pad_width, mode='constant', constant_values=0)
        elif T_loss > T_rnn:
            dout = dout[:, :T_rnn, :]

        # --- ✨ RNN Backward (dout만 반환받음) ✨ ---
        layers_rnn = list(self.rnn_layers.values())
        layers_rnn.reverse()
        for layer in layers_rnn:
            dout = layer.backward(dout) # 출력: (N, T_rnn, D)
        # ------------------------------------------

        # CNN 입력 형태로 변환
        N, T, D = dout.shape
        last_conv_key_index = len([k for k in self.cnn_layers if k.startswith('Conv')])
        C = self.params[f'W{last_conv_key_index}'].shape[0]
        H = D // C
        dout = dout.reshape(N, T, C, H).transpose(0, 2, 3, 1)

        # --- ✨ CNN Backward (dout만 반환받음) ✨ ---
        layers_cnn = list(self.cnn_layers.values())
        layers_cnn.reverse()
        for layer in layers_cnn:
            dout = layer.backward(dout)
        # ------------------------------------------

        # --- ✨ 결과 저장 (각 계층의 속성에서 가져옴) ✨ ---
        grads = {}
        cnn_layer_idx = 1
        num_conv_layers = len([k for k in self.cnn_layers if k.startswith('Conv')])

        for i in range(num_conv_layers):
             conv_key = f'Conv{cnn_layer_idx}'
             bn_key = f'BN{cnn_layer_idx}'
             # 각 계층 객체에서 직접 그래디언트 속성 가져오기
             grads[f'W{cnn_layer_idx}'] = self.cnn_layers[conv_key].dW
             grads[f'b{cnn_layer_idx}'] = self.cnn_layers[conv_key].db
             grads[f'gamma{cnn_layer_idx}'] = self.cnn_layers[bn_key].dgamma
             grads[f'beta{cnn_layer_idx}'] = self.cnn_layers[bn_key].dbeta
             cnn_layer_idx += 1

        # RNN 계층 객체에서 직접 grads 속성 가져오기
        grads['LSTM_Wx'], grads['LSTM_Wh'], grads['LSTM_b'] = self.rnn_layers['LSTM'].grads
        grads['TimeAffine_W'], grads['TimeAffine_b'] = self.rnn_layers['TimeAffine'].grads
        # ----------------------------------------------

        return grads
    # ------------------------------------