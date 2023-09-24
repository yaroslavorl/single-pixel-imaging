from camera import SinglePixelCamera
from utils import *

import cv2
import torch


def preprocess(img, pattern_num):

    img = torch.FloatTensor(img) / 255.0
    size_img = img.shape

    basis = cosine_transform_matrix(size_img)
    binary_pattern = binary_mask(size_img, pattern_num)
    output = single_point_detector(binary_pattern, img.flatten())

    intended_img = torch.randint(0, 2, (size_img[0], size_img[1]), dtype=torch.float32, device=device)
    intended_img = torch.matmul(basis.to(device), intended_img.flatten())
    intended_img.requires_grad = True

    transposed_basis = basis.T
    P = torch.matmul(binary_pattern, transposed_basis)

    return intended_img, output, P, transposed_basis


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    input_img = cv2.imread('single-pixel-imaging/img/lena64.jpeg', 0)
    intended_img, output, P, transposed_basis = preprocess(input_img, pattern_num=2000)

    model = SinglePixelCamera(intended_img, loss=torch.nn.MSELoss(), optimizer=torch.optim.AdamW, lr=0.03)
    model.fit(P.to(device), output.to(device), epochs=1000)

    img = model.get_img()
    cv2.imwrite('single-pixel-imaging/img/test_2.jpeg', torch.matmul(transposed_basis, img.to('cpu')).reshape(input_img.shape).numpy() * 255)
