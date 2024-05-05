import torch
from tqdm import tqdm


class SinglePixelCamera:

    def __init__(self, img, loss, optimizer, save_log: bool, frequency_saving: int = 10):
        self._img = img
        self._loss = loss
        self._optimizer = optimizer
        self.loss_list = []

        self.save_log = save_log
        self.frequency_saving = frequency_saving
        self.log_list = []

    def fit(self, P, y, epochs):
        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            output = torch.matmul(P, self._img)
            error = self._loss(output, y) + self._l1_reg(self._img)

            self._optimizer.zero_grad()
            error.backward()
            self.loss_list.append(error.item())
            self._optimizer.step()

            if self.save_log and epoch % self.frequency_saving == 0:
                self.log_list.append(self.get_img)

    @property
    def get_img(self):
        return self._img.detach().clone()

    @staticmethod
    def _l1_reg(w, alpha=0.1):
        return alpha * torch.sum(torch.abs(w))
