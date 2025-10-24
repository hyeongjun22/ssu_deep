# layers.py
import cupy as cp
from .cu_util import im2col, col2im

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = cp.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = cp.dot(dout, self.W.T)
        self.dW = cp.dot(self.x.T, dout)
        self.db = cp.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)
        return dx

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        self.col = im2col(x, FH, FW, self.stride, self.pad)
        self.col_W = self.W.reshape(FN, -1).T
        out = cp.dot(self.col, self.col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = cp.sum(dout, axis=0)
        self.dW = cp.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = cp.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx

class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        self.arg_max = cp.argmax(col, axis=1)
        out = cp.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dmax = cp.zeros((dout.size, pool_size), dtype=dout.dtype)
        dmax[cp.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = cp.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma # 스케일 파라미터 (학습 대상)
        self.beta = beta   # 시프트 파라미터 (학습 대상)
        self.momentum = momentum
        self.input_shape = None # (N, C, H, W) 또는 (N, D)

        # 추론 시 사용할 이동 평균
        self.running_mean = running_mean
        self.running_var = running_var

        # 역전파 시 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
        self.xn = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        
        # Affine 계층 (2D)
        if x.ndim == 2:
            N, D = x.shape
            if self.running_mean is None:
                self.running_mean = cp.zeros(D, dtype=x.dtype)
                self.running_var = cp.zeros(D, dtype=x.dtype)
            
            if train_flg:
                mu = cp.mean(x, axis=0)
                xc = x - mu
                var = cp.mean(xc**2, axis=0)
                std = cp.sqrt(var + 10e-7)
                xn = xc / std

                self.batch_size = N
                self.xc = xc
                self.xn = xn
                self.std = std
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            else:
                xc = x - self.running_mean
                xn = xc / (cp.sqrt(self.running_var + 10e-7))
                
            out = self.gamma * xn + self.beta 
        
        # Conv 계층 (4D)
        elif x.ndim == 4:
            N, C, H, W = x.shape
            if self.running_mean is None:
                self.running_mean = cp.zeros(C, dtype=x.dtype)
                self.running_var = cp.zeros(C, dtype=x.dtype)

            if train_flg:
                # 채널(C)별로 평균/분산 계산 (N, H, W 축에 대해)
                mu = cp.mean(x, axis=(0, 2, 3)) # (C,)
                xc = x - mu.reshape(1, C, 1, 1) # 브로드캐스팅
                var = cp.mean(xc**2, axis=(0, 2, 3)) # (C,)
                std = cp.sqrt(var + 10e-7) # (C,)
                xn = xc / std.reshape(1, C, 1, 1) # 브로드캐스팅
                
                self.batch_size = N
                self.xc = xc
                self.xn = xn
                self.std = std
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            else:
                mu = self.running_mean.reshape(1, C, 1, 1)
                var = self.running_var.reshape(1, C, 1, 1)
                xn = (x - mu) / cp.sqrt(var + 10e-7)
                
            out = self.gamma.reshape(1, C, 1, 1) * xn + self.beta.reshape(1, C, 1, 1)
        
        return out

    def backward(self, dout):
        # Affine 계층 (2D)
        if dout.ndim == 2:
            N, D = dout.shape
            self.dbeta = cp.sum(dout, axis=0)
            self.dgamma = cp.sum(self.xn * dout, axis=0)
            
            dxn = self.gamma * dout
            dxc = dxn / self.std
            dstd = -cp.sum((dxn * self.xc) / (self.std * self.std), axis=0)
            dvar = 0.5 * dstd / self.std
            dxc += (2.0 / self.batch_size) * self.xc * dvar
            dmu = -cp.sum(dxc, axis=0)
            dx = dxc + dmu / self.batch_size

        # Conv 계층 (4D)
        elif dout.ndim == 4:
            N, C, H, W = dout.shape
            
            # (N, H, W) 축에 대해 합산
            self.dbeta = cp.sum(dout, axis=(0, 2, 3))
            self.dgamma = cp.sum(self.xn * dout, axis=(0, 2, 3))
            
            gamma_rs = self.gamma.reshape(1, C, 1, 1)
            std_rs = self.std.reshape(1, C, 1, 1)
            
            dxn = gamma_rs * dout
            dxc = dxn / std_rs
            dstd = -cp.sum((dxn * self.xc) / (std_rs * std_rs), axis=(0, 2, 3))
            dvar = 0.5 * dstd / self.std
            dvar_rs = dvar.reshape(1, C, 1, 1)
            
            # 평균 계산 시 N*H*W로 나누었으므로 역전파 시에도 반영
            dxc += (2.0 / (N * H * W)) * self.xc * dvar_rs 
            dmu = -cp.sum(dxc, axis=(0, 2, 3))
            dmu_rs = dmu.reshape(1, C, 1, 1)
            dx = dxc + dmu_rs / (N * H * W)

        return dx

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - cp.max(x, axis=0)
        y = cp.exp(x) / cp.sum(cp.exp(x), axis=0)
        return y.T
    x = x - cp.max(x)
    return cp.exp(x) / cp.sum(cp.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -cp.sum(cp.log(y[cp.arange(batch_size), t] + 1e-7)) / batch_size



def sigmoid(x):
    return 1 / (1 + cp.exp(-x))

class LSTM:
    """
    LSTM 셀 클래스 (TimeLSTM이 사용합니다)
    """
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [cp.zeros_like(Wx), cp.zeros_like(Wh), cp.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        # 아핀 변환
        A = cp.dot(x, Wx) + cp.dot(h_prev, Wh) + b

        # 게이트 분리 (f, g, i, o)
        f = sigmoid(A[:, :H])
        g = cp.tanh(A[:, H:2*H])
        i = sigmoid(A[:, 2*H:3*H])
        o = sigmoid(A[:, 3*H:])

        # 셀 상태 및 은닉 상태 계산
        c_next = (f * c_prev) + (i * g)
        h_next = o * cp.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = cp.tanh(c_next)
        
        ds = dc_next + (dh_next * o) * (1 - tanh_c_next**2)
        
        dc_prev = ds * f

        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i

        # 시그모이드/tanh 역전파
        di_input = di * i * (1 - i)
        df_input = df * f * (1 - f)
        do_input = do * o * (1 - o)
        dg_input = dg * (1 - g**2)

        # 아핀 변환 역전파
        dA = cp.hstack((df_input, dg_input, di_input, do_input))
        
        dWh = cp.dot(h_prev.T, dA)
        dWx = cp.dot(x.T, dA)
        db = cp.sum(dA, axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = cp.dot(dA, Wx.T)
        dh_prev = cp.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev



class TimeLSTM:
    """
    TimeLSTM 계층 (cu_layers.py에 이미 있을 수 있지만, 완전한 구성을 위해 포함)
    """
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [cp.zeros_like(Wx), cp.zeros_like(Wh), cp.zeros_like(b)]
        self.layers = None
        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = cp.empty((N, T, H), dtype=xs.dtype)

        if not self.stateful or self.h is None:
            self.h = cp.zeros((N, H), dtype=xs.dtype)
        if not self.stateful or self.c is None:
            self.c = cp.zeros((N, H), dtype=xs.dtype)

        for t in range(T):
            layer = LSTM(Wx, Wh, b) # LSTM 셀 (cu_layers.py에 정의되어 있다고 가정)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = cp.empty((N, T, D), dtype=dhs.dtype)
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None

class TimeAffine:
    """
    TimeAffine 계층 (cu_layers.py에 이미 있을 수 있지만, 완전한 구성을 위해 포함)
    """
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [cp.zeros_like(W), cp.zeros_like(b)]
        self.x = None

    def forward(self, xs):
        N, T, D = xs.shape
        W, b = self.params

        rx = xs.reshape(N*T, -1)
        out = cp.dot(rx, W) + b
        self.x = xs # (N, T, D)
        return out.reshape(N, T, -1) # (N, T, V)

    def backward(self, dout):
        N, T, V = dout.shape
        W, b = self.params

        dout = dout.reshape(N*T, -1) # (N*T, V)
        rx = self.x.reshape(N*T, -1) # (N*T, D)

        db = cp.sum(dout, axis=0)
        dW = cp.dot(rx.T, dout)
        dx = cp.dot(dout, W.T)
        dx = dx.reshape(*self.x.shape) # (N, T, D)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class CTCLoss:
    """
    Connectionist Temporal Classification (CTC) Loss (CuPy 구현)
    구현이 매우 복잡하며, 여기서는 핵심 로직을 제공합니다.
    (참고: 실제 구현은 수치 안정성을 위해 log-space에서 수행되어야 함)
    """
    def __init__(self, blank_id=0):
        self.blank_id = blank_id
        self.y_softmax = None
        self.t_ext = None
        self.y_len = None
        self.t_len = None

    def _prepare_labels(self, t, N, S_max):
        """ 레이블을 (N, 2*S_max + 1) 형태로 변형 (blank 삽입) """
        t_ext = cp.full((N, 2 * S_max + 1), self.blank_id, dtype=t.dtype)
        t_ext[:, 1::2] = t
        return t_ext

    def forward(self, y_pred, t, y_len, t_len):
        """
        y_pred: (N, T, V+1) - 모델의 raw 출력 (softmax 전)
        t: (N, S_max) - 패딩된 정답 레이블
        y_len: (N,) - 각 y_pred의 실제 시퀀스 길이 (T)
        t_len: (N,) - 각 t의 실제 레이블 길이 (S)
        """
        N, T, V = y_pred.shape
        S_max = t.shape[1]

        # 1. Softmax 적용 (수치 안정성을 위해 log_softmax가 더 좋음)
        y_pred_exp = cp.exp(y_pred - cp.max(y_pred, axis=2, keepdims=True))
        self.y_softmax = y_pred_exp / cp.sum(y_pred_exp, axis=2, keepdims=True)
        
        # 2. 레이블 확장 (blank 삽입)
        self.t_ext = self._prepare_labels(t, N, S_max)
        self.y_len = y_len
        self.t_len = t_len
        
        S_ext_max = 2 * S_max + 1
        
        # 3. Forward-DP (alpha)
        # alpha[n, t, s] = t 시점, 레이블 s까지의 확률
        alpha = cp.zeros((N, T, S_ext_max), dtype=cp.float32)

        # 초기화 (t=0)
        alpha[:, 0, 0] = self.y_softmax[:, 0, self.blank_id]
        alpha[:, 0, 1] = self.y_softmax[cp.arange(N), 0, self.t_ext[:, 1]]

        # DP
        for t in range(1, T):
            for s in range(S_ext_max):
                # t_ext[n, s]에 해당하는 글자의 인덱스
                char_idx = self.t_ext[cp.arange(N), s]
                
                # y_softmax[n, t, char_idx]
                prob = self.y_softmax[cp.arange(N), t, char_idx]
                
                # case 1: 이전 s에서 그대로
                a_prev = alpha[:, t-1, s]
                
                # case 2: 이전 s-1에서
                if s > 0:
                    a_prev += alpha[:, t-1, s-1]
                
                # case 3: 이전 s-2에서 (반복 방지)
                if s > 1 and self.t_ext[0, s] != self.blank_id and self.t_ext[0, s] != self.t_ext[0, s-2]:
                     a_prev += alpha[:, t-1, s-2]
                
                alpha[:, t, s] = prob * a_prev

        # 4. Loss 계산
        total_loss = cp.zeros(N, dtype=cp.float32)
        for n in range(N):
            L = self.t_len[n] * 2 + 1
            T_n = self.y_len[n]
            # t_len[n]에 해당하는 마지막 글자(L-1)와 마지막 blank(L-2)의 확률
            final_prob = alpha[n, T_n-1, L-1] + alpha[n, T_n-1, L-2]
            total_loss[n] = -cp.log(final_prob + 1e-9) # log-loss

        self.cache = (alpha,) # backward를 위해 저장
        self.loss = cp.mean(total_loss) # ✨ loss를 속성에 저장
        return self.loss

    def backward(self, dout=1):
        N, T, V = self.y_softmax.shape
        S_ext_max = self.t_ext.shape[1]
        alpha = self.cache[0]
        
        # 1. Backward-DP (beta)
        beta = cp.zeros_like(alpha)
        
        # 초기화 (t=T-1)
        for n in range(N):
            L = self.t_len[n] * 2 + 1
            T_n = self.y_len[n]
            beta[n, T_n-1, L-1] = 1.0
            beta[n, T_n-1, L-2] = 1.0

        # DP
        for t in reversed(range(T - 1)):
            for s in range(S_ext_max):
                char_idx = self.t_ext[cp.arange(N), s]
                prob = self.y_softmax[cp.arange(N), t+1, char_idx]

                b_next = beta[:, t+1, s]
                if s < S_ext_max - 1:
                    b_next += beta[:, t+1, s+1]
                if s < S_ext_max - 2 and self.t_ext[0, s] != self.blank_id and self.t_ext[0, s] != self.t_ext[0, s+2]:
                    b_next += beta[:, t+1, s+2]
                    
                beta[:, t, s] = prob * b_next

        # 2. 그래디언트 계산 (dy)
        dy_softmax = cp.zeros_like(self.y_softmax)
        
        # alpha * beta
        ab = alpha * beta
        
        for t in range(T):
            for s in range(S_ext_max):
                char_idx = self.t_ext[cp.arange(N), s]
                
                # (N,)
                prob = ab[:, t, s] 
                
                # CuPy는 += 연산이 느리므로, cp.add.at을 사용
                # dy_softmax[n, t, char_idx] += prob
                cp.add.at(dy_softmax, (cp.arange(N), t, char_idx), prob)

        # y_softmax (N, T, V)
        # P(L|X) = sum_s(alpha_t(s) * beta_t(s))
        P_L_X = cp.sum(ab, axis=2, keepdims=True) # (N, T, 1)
        
        # dL/dy_k = y_k - (1/P(L|X)) * sum(s|l_s=k) alpha_t(s) * beta_t(s)
        dy = self.y_softmax - (dy_softmax / (P_L_X + 1e-9))
        
        # 정규화 및 dout 적용
        dy /= N
        return dy * dout

 
class TimeSoftmaxWithLossMasked:
    """
    TimeSoftmaxWithLoss에 PAD 토큰 마스킹 기능을 추가한 버전.
    정답 레이블이 PAD인 위치에서는 손실과 그래디언트를 계산하지 않습니다.
    """
    def __init__(self, pad_id=0):
        self.params, self.grads = [], []
        self.cache = None
        self.pad_id = pad_id # <PAD> 토큰의 ID

    def forward(self, xs, ts):
        """
        xs: 모델 출력 (N, T, V)
        ts: 정답 레이블 (N, T)
        """
        N, T, V = xs.shape

        # 마스크 생성: 정답(ts)이 PAD가 아닌 위치는 True(1), PAD인 위치는 False(0)
        mask = (ts != self.pad_id)
        num_valid_steps = cp.sum(mask) # 실제 손실 계산에 사용될 타임 스텝 총 개수

        # one-hot 변환 (PAD 위치도 일단 변환)
        ts_flat = ts.astype(cp.int32).flatten()
        # Ensure indices are within bounds before creating one-hot encoding
        ts_flat_clipped = cp.clip(ts_flat, 0, V - 1)
        self.ts_one_hot = cp.eye(V, dtype=xs.dtype)[ts_flat_clipped]
        # Handle potential out-of-bounds indices if needed, though clipping is safer
        # if cp.any(ts_flat >= V) or cp.any(ts_flat < 0):
        #     print("Warning: Label indices out of vocab size range detected.")


        # Softmax 계산
        ys = softmax(xs.reshape(-1, V)).reshape(N, T, V)

        # 모든 위치에 대한 Cross Entropy 계산 (element-wise)
        loss_elements = -self.ts_one_hot * cp.log(ys.reshape(-1, V) + 1e-7)
        loss_per_element_summed = cp.sum(loss_elements, axis=1) # V 차원에 대해 합산, shape: (N*T,)
        loss_per_step = loss_per_element_summed.reshape(N, T) # shape: (N, T)

        # 마스크 적용: PAD 위치의 손실을 0으로 만듦
        masked_loss_per_step = loss_per_step * mask

        # 전체 배치에서 유효한 스텝들의 손실 합계 계산
        total_masked_loss = cp.sum(masked_loss_per_step)

        # 유효한 스텝들의 개수로 나누어 평균 손실 계산
        avg_loss = total_masked_loss / (num_valid_steps + 1e-7)

        self.cache = (ts, ys, mask) # ✨ 마스크도 캐시에 저장
        return avg_loss

    def backward(self, dout=1):
        ts, ys, mask = self.cache # ✨ 마스크 로드
        N, T, V = ys.shape

        # 그래디언트 계산 (ys - t_one_hot)
        # Ensure ts_one_hot is correctly shaped if loaded from cache
        if self.ts_one_hot.shape[0] != N * T:
             ts_flat = ts.astype(cp.int32).flatten()
             ts_flat_clipped = cp.clip(ts_flat, 0, V - 1)
             self.ts_one_hot = cp.eye(V, dtype=ys.dtype)[ts_flat_clipped]


        dx = ys.reshape(-1, V) - self.ts_one_hot
        dx = dx.reshape(N, T, V) # 원래 shape으로 복원

        # 마스크 적용: PAD 위치의 그래디언트를 0으로 만듦
        dx_masked = dx * mask[:, :, None]

        # 정규화 (배치 크기로 나눔) 및 dout 곱하기
        dx_final = dx_masked * dout / N

        return dx_final       
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[cp.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx

class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, xs, ts):
        N, T, V = xs.shape
        self.ts_one_hot = cp.eye(V, dtype=xs.dtype)[ts.reshape(-1)]
        
        ys = softmax(xs.reshape(-1, V)).reshape(N, T, V)
        loss = -cp.sum(self.ts_one_hot * cp.log(ys.reshape(-1, V) + 1e-7)) / N
        
        self.cache = (ts, ys)
        return loss

    def backward(self, dout=1):
        ts, ys = self.cache
        N, T, V = ys.shape
        
        dx = ys.reshape(-1, V) - self.ts_one_hot
        dx *= dout / N
        dx = dx.reshape(N, T, V)
        
        return dx