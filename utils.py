import numpy as np
import torch
from scipy.fft import dct


def single_point_detector(pattern, img):
    return torch.matmul(pattern, img)


def cosine_transform_matrix(size_img):
    return torch.FloatTensor(dct(np.eye(size_img[0] * size_img[1]), axis=0, norm='ortho'))


def binary_mask(size_img, pattern_num):
    return 2 * torch.randint(0, 2, size=(pattern_num, size_img[0] * size_img[1]), dtype=torch.float32) - 1
