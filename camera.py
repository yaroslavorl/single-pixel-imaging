import torch


class SinglePixelCamera:

    def __init__(self, img, loss, optimizer, lr):

        self.img = img
        self.loss = loss
        self.optimizer = optimizer([self.img], lr=lr)

    @staticmethod
    def lasso(w, alpha=0.1):
        return alpha * torch.sum(torch.abs(w))

    def get_img(self):
        return self.img.detach().clone()

    def fit(self, P, y, epochs):

        for epoch in range(1, epochs + 1):
            output = torch.matmul(P, self.img)
            error = self.loss(output, y) + self.lasso(self.img)

            self.optimizer.zero_grad()
            error.backward()
            self.optimizer.step()

            print(f'Эпоха: {epoch}/{epochs}, loss: {error.item()}')
            