import torch
from tqdm import tqdm


class SinglePixelCamera:

    def __init__(self, img, loss, optimizer, history: bool, frequency_saving: int = 10):
        self._img = img
        self._loss = loss
        self.loss_list = []
        self._optimizer = optimizer

        self.history = history
        self.frequency_saving = frequency_saving
        self.img_history_list = []

    def fit(self, P, y, epochs):
        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            output = torch.matmul(P, self._img)
            error = self._loss(output, y) + self._l1_reg(self._img)

            self._optimizer.zero_grad()
            error.backward()
            self.loss_list.append(error.item())
            self._optimizer.step()

            if self.history and epoch % self.frequency_saving == 0:
                self.img_history_list.append(self.get_img)

    @property
    def get_img(self):
        return self._img.detach().clone()

    @staticmethod
    def _l1_reg(w, alpha=0.1):
        return alpha * torch.norm(w, 1)
