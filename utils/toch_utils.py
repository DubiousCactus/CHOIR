import numpy as np
import torch






def random_rotate_np(x):
    aa = np.random.randn(3)
    theta = np.sqrt(np.sum(aa**2))
    k = aa / np.maximum(theta, 1e-6)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.matmul(K, K)
    R = R.astype(np.float32)
    return np.matmul(x, R), R






