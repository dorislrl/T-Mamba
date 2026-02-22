# -*- coding:utf-8 -*-
import numpy, pdb
import torch
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np

import iisignature
from fastdtw import fastdtw as dtw #https://github.com/slaypni/fastdtw/issues

def diff(x):
    dx = numpy.convolve(x, [0.5,0,-0.5], mode='same'); dx[0] = dx[1]; dx[-1] = dx[-2]
    # dx = numpy.convolve(x, [0.2,0.1,0,-0.1,-0.2], mode='same'); dx[0] = dx[1] = dx[2]; dx[-1] = dx[-2] = dx[-3]
    return dx

def diffTheta(x):
    dx = numpy.zeros_like(x)
    dx[1:-1] = x[2:] - x[0:-2]; dx[-1] = dx[-2]; dx[0] = dx[1]
    temp = numpy.where(numpy.abs(dx)>numpy.pi)
    dx[temp] -= numpy.sign(dx[temp]) * 2 * numpy.pi
    dx *= 0.5
    return dx
class butterLPFilter(object):
    """docstring for butterLPFilter"""
    def __init__(self, highcut=10.0, fs=200.0, order=3):
        super(butterLPFilter, self).__init__()
        nyq = 0.5 * fs
        highcut = highcut / nyq
        b, a = signal.butter(order, highcut, btype='low')
        self.b = b
        self.a = a
    def __call__(self, data):
        y = signal.filtfilt(self.b, self.a, data) 
        return y

bf = butterLPFilter(15, 100)

def leadlag_transform(x):

    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None] 
    x_rep = np.repeat(x, 2, axis=0)
    lead = x_rep[1:]
    lag = x_rep[:-1]
    ll_path = np.hstack((lead, lag))
    return ll_path
from sklearn.base import BaseEstimator, TransformerMixin
class AddTime(BaseEstimator, TransformerMixin):
    """Add time component to a single path of shape [L, C]."""
    def fit(self, data, labels=None):
        return self

    def transform(self, data):
        """
        data: numpy.ndarray or torch.Tensor of shape [L, C]
        returns: numpy.ndarray of shape [L, C+1]
        """
        # 转成 torch.Tensor
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        L, C = data.shape

        # 时间通道 [0, 1]，长度为 L
        time_scaled = torch.linspace(0, 1, L, device=data.device).view(L, 1)

        # 拼接在第一列
        out = torch.cat((time_scaled, data), dim=1)

        return out.cpu().numpy()
class AppendZero(BaseEstimator, TransformerMixin):
    """Append a zero starting vector to every path.
    Supports both (L,C) and (B,L,C) inputs.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 如果是 numpy，转成 torch
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        # 如果是 (L,C)，扩展 batch 维度
        added_batch = False
        if X.dim() == 2:  # (L,C)
            X = X.unsqueeze(0)  # -> (1,L,C)
            added_batch = True

        B, L, C = X.shape
        zero_vec = torch.zeros(size=(B, 1, C), device=X.device, dtype=X.dtype)
        X_out = torch.cat((zero_vec, X), dim=1)

        # 如果原来是 (L,C)，去掉 batch 维度
        if added_batch:
            X_out = X_out.squeeze(0)  # -> (L+1,C)

        return X_out.cpu().numpy()    

class ShiftToZero:
    """Performs a translation so paths begin at zero (NumPy version).
    Supports both (L, C) and (B, L, C) input.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(X)}")

        if X.ndim == 2:
            # (L, C)
            return X - X[0:1, :]
        elif X.ndim == 3:
            # (B, L, C)
            return X - X[:, [0], :]
        else:
            raise ValueError(f"Unsupported input shape: {X.shape}")

    def fit_transform(self, X, y=None):
        """Combine fit and transform."""
        self.fit(X, y)
        return self.transform(X)

def sigfeatExt(pathList, feats, dim=2, 
               transform=False, finger_scene=False,
               window_size=10, stride=1, signature_depth=2,
               use_leadlag=False, use_logsig=False, 
                 use_dyadic=False, dyadic_depth=5):
    """
    输入：
        pathList: list[np.ndarray] or list[torch.Tensor] (L,C)
        feats: list 收集特征
    """
    for path in pathList:
        if isinstance(path, torch.Tensor):
            path = path.cpu().numpy()

        p = path[:, dim]  
        path[:, 0] = bf(path[:, 0]) 
        path[:, 1] = bf(path[:, 1])  
        total_len = np.sum(np.sqrt(np.sum(np.diff(path[:, 0:2], axis=0)**2, axis=1))) + 1e-6
        path[:, 0:2] /= total_len
        dx = diff(path[:, 0]); dy = diff(path[:, 1])
        v = numpy.sqrt(dx**2+dy**2)
        theta = numpy.arctan2(dy, dx)
        cos = numpy.cos(theta)
        sin = numpy.sin(theta)
        dv = diff(v)
        dtheta = numpy.abs(diffTheta(theta))
        logCurRadius = numpy.log((v+0.05) / (dtheta+0.05))
        dv2 = numpy.abs(v*dtheta)
        totalAccel = numpy.sqrt(dv**2 + dv2**2)
        dynamic_feat = np.column_stack((
            dx, dy, v, cos, sin,
            theta, logCurRadius, totalAccel,
            dv, dv2, dtheta, p
        )).astype(np.float32)


        if finger_scene:
            ''' For finger scenario '''
            dynamic_feat[:,:-1] = (dynamic_feat[:,:-1] - numpy.mean(dynamic_feat[:,:-1], axis=0)) / numpy.std(dynamic_feat[:,:-1], axis=0)
        else:
            ''' For stylus scenario '''
            dynamic_feat = (dynamic_feat - numpy.mean(dynamic_feat, axis=0)) / numpy.std(dynamic_feat, axis=0)

        transformer1 = AppendZero()
        dynamic_feat = transformer1.fit_transform(dynamic_feat)  # -> (L+1,C)

        transformer2 = AddTime()
        dynamic_feat = transformer2.fit_transform(dynamic_feat)  # -> (L+1,C+1)

        signatures = []
        if use_dyadic:
            dy_segments = dyadic_windows_segments(torch.from_numpy(dynamic_feat), depth=dyadic_depth)
            for d, segs in dy_segments.items():
                for seg in segs:
                    if use_logsig:
                        sig = iisignature.logsig(seg.numpy(), iisignature.prepare(seg.shape[1], signature_depth))
                    else:
                        sig = iisignature.sig(seg.numpy(), signature_depth)
                    signatures.append(sig)
        else: 
            num_samples = dynamic_feat.shape[0]
            d_dyn = dynamic_feat.shape[1]
            prep_dyn = iisignature.prepare(d_dyn, signature_depth)

            for start in range(0, num_samples - window_size + 1, stride):
                window_dyn = dynamic_feat[start:start + window_size, :]
                if window_dyn.shape[0] == 0 or window_dyn.ndim < 2:
                    continue
                if use_logsig:
                    sig_full = iisignature.logsig(window_dyn, prep_dyn)
                else:
                    sig_full = iisignature.sig(window_dyn, signature_depth)

                signatures.append(sig_full)

        signatures = np.array(signatures, dtype=np.float64)
        feats.append(signatures)

    return feats
