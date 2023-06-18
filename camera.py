import torch
import cv2


class Camera:

    def __init__(self, size_img, pattern_num):

        self.img = None
        self.size_img = size_img
        self.pattern_num = pattern_num
        self.loss = None
        self.optimizer = None

    def detector(self, pattern, img):
        return torch.matmul(pattern, img)

    def dct_matrix(self):
        n = self.size_img[0] * self.size_img[1]
        matrix = torch.zeros((n, n))
        for i in range(1, n):
            for j in range(n):
                matrix[i][j] = (2 * j + 1) * i * torch.pi / (2 * n)
        matrix = torch.sqrt(torch.tensor(2 / n)) * torch.cos(matrix)
        matrix[0] = 1 / torch.sqrt(torch.tensor(n))

        return matrix

    def binary_mask(self):
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

    def fit(self, basis, y, epochs):

        for epoch in range(1, epochs + 1):
            output = torch.matmul(basis, self.img)
            error = self.loss(output, y) + self.lasso(self.img)

            self.optimizer.zero_grad()
            error.backward()
            self.optimizer.step()

            print(f'Эпоха: {epoch}/{epochs}, loss: {error.item()}')


if __name__ == '__main__':

    pattern_num = 300
    img = cv2.imread('test.jpg', 0)
    img = torch.FloatTensor(img) / 255
    size_img = img.shape

    model = Camera(size_img=size_img, pattern_num=pattern_num)

    basis = model.dct_matrix()
    pattern = model.binary_mask()
    y = model.detector(pattern, img.flatten())

    img = torch.randint(0, 2, (size_img[0], size_img[1]), dtype=torch.float32)

    img = torch.matmul(basis, img.flatten())
    basis = basis.T
    p = torch.matmul(pattern, basis)

    img.requires_grad = True
    model.compile(img, loss=torch.nn.MSELoss, optimizer=torch.optim.Adam, lr=0.03)
    model.fit(p, y, epochs=500)

    img = model.get_img()

    cv2.imwrite('test_2.jpg',
                torch.matmul(basis, img).reshape(size_img).numpy() * 255)
