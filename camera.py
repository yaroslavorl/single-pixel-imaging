import torch
import cv2
import time


def detector(x, pattern):
    return torch.matmul(pattern, x)


def dct_matrix(n):
    matrix = torch.zeros((n, n))
    
    for i in range(1, n):
        for j in range(n):
            matrix[i][j] = (2 * j + 1) * i * torch.pi / (2 * n)

    matrix = torch.sqrt(torch.tensor(2 / n)) * torch.cos(matrix)
    matrix[0] = 1 / torch.sqrt(torch.tensor(n))

    return matrix


def binary_mask(size):
    return 2 * torch.randint(0, 2, size, dtype=torch.float32) - 1


def lasso(w, alpha=0.1):
    return alpha * torch.sum(torch.abs(w))


def grad_descent(img, img_y, p, loss, optimizer, epochs):

    for epoch in range(1, epochs + 1):

        output = torch.matmul(p, img)
        error = loss(output, img_y) + lasso(img)

        optimizer.zero_grad()
        error.backward()
        optimizer.step()

        print(f'Эпоха: {epoch}/{epochs}, loss: {error.item()}')

    return img


if __name__ == "__main__":

    pattern_num = 300

    img = cv2.imread('/home/jaroslav/PycharmProjects/dcan/single_pixel_camera/img/test_cs_32.png', 0)
    img = torch.FloatTensor(img) / 255
    size_img = img.shape

    basis = dct_matrix(size_img[0] * size_img[1])
    pattern = binary_mask(size=(pattern_num, size_img[0] * size_img[1]))
    y = detector(img.flatten(), pattern)

    img = torch.randint(0, 2, (size_img[0], size_img[1]), dtype=torch.float32)
    img = torch.matmul(basis, img.flatten())
    basis = basis.T
    p = torch.matmul(pattern, basis)

    img.requires_grad = True
    optimizer = torch.optim.Adam([img], lr=0.03)
    loss = torch.nn.MSELoss()

    start = time.time()
    img = grad_descent(img, y, p, loss, optimizer, epochs=500)
    print(time.time() - start)

    cv2.imwrite('/home/jaroslav/PycharmProjects/dcan/single_pixel_camera/img/test.jpg', torch.matmul(basis, img).reshape(size_img).detach().numpy() * 255)
