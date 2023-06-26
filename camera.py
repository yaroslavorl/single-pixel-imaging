import torch
import numpy as np
from scipy.fft import dct


class Camera:

    def __init__(self, size_img, pattern_num):

        self.img = None
        self.size_img = size_img
        self.pattern_num = pattern_num
        self.loss = None
        self.optimizer = None

    def detector(self, pattern, img):
        return torch.matmul(pattern, img)

    def get_dct_matrix(self):
        return torch.FloatTensor(dct(np.eye(self.size_img[0] * self.size_img[1]), axis=0, norm='ortho'))

    def get_binary_mask(self):
        return 2 * torch.randint(0, 2, size=(self.pattern_num, self.size_img[0] * self.size_img[1]),
                                 dtype=torch.float32) - 1

    def get_img(self):
        return self.img.detach().clone()

    def lasso(self, w, alpha=0.1):
        return alpha * torch.sum(torch.abs(w))

    def compile(self, img, loss, optimizer, lr):
        self.img = img
        self.loss = loss()
        self.optimizer = optimizer([self.img], lr=lr)

    def fit(self, p, y, epochs):

        for epoch in range(1, epochs + 1):
            output = torch.matmul(p, self.img)
            error = self.loss(output, y) + self.lasso(self.img)

            self.optimizer.zero_grad()
            error.backward()
            self.optimizer.step()

            print(f'Эпоха: {epoch}/{epochs}, loss: {error.item()}')
            