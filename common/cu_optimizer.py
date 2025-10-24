# cu_optimizer.py
import cupy as cp
import numpy as np # lr_t 계산에만 일시적으로 사용

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = cp.zeros_like(val)
                self.v[key] = cp.zeros_like(val)
        
        self.iter += 1
        # lr_t 계산은 스칼라 값이므로 NumPy를 사용해도 무방합니다.
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            # 모든 연산이 CuPy 배열 간에 이루어집니다.
            
            # 1. 모멘텀(m) 업데이트
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            
            # 2. 분산(v) 업데이트
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            # 3. ✨✨✨ 누락된 파라미터 업데이트 ✨✨✨
            # 계산된 m과 v를 사용하여 params[key]를 갱신합니다.
            params[key] -= lr_t * self.m[key] / (cp.sqrt(self.v[key]) + 1e-7)