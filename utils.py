from typing import Tuple

import cv2
import numpy as np
import torch
from scipy.fft import dct


def single_point_detector(pattern, img):
    return torch.matmul(pattern, img)


def dct_cosine_transform_matrix(img_size: Tuple[int, int]):
    return torch.from_numpy(dct(np.eye(img_size[0] * img_size[1], dtype=np.float32), axis=0, norm='ortho'))


def dct_img_to_pixels(img, tr_basis, img_size: Tuple[int, int]):
    return torch.matmul(tr_basis, img.to('cpu')).view(img_size).numpy() * 255


def binary_mask(img_size: Tuple[int, int], pattern_num):
    return 2 * torch.randint(0, 2, size=(pattern_num, img_size[0] * img_size[1]), dtype=torch.float32) - 1


cv2.imwrite('img/mr_president.jpg', cv2.resize(cv2.imread('/home/yaroslav/Загрузки/mr_president.jpg', 0), (128, 128)))