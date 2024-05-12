import os

import torch
import cv2

from src.camera import SinglePixelCamera
from src.utils import single_point_detector, dct_cosine_transform_matrix, dct_img_to_pixels, binary_mask


def preprocess(img, num_patterns, device):
    img = torch.from_numpy(img) / 255.0
    size_img = img.size()

    # TODO: discrete cosine transform basis
    basis = dct_cosine_transform_matrix(size_img)
    binary_pattern = binary_mask(size_img, num_patterns)
    # TODO: camera equation Px = y, where P - binary patterns, x - image, y - multiplication result
    output_y = single_point_detector(binary_pattern, img.flatten())

    initial_img = torch.randint(0, 2, (size_img[0], size_img[1]), dtype=torch.float32, device=device)
    initial_img = torch.matmul(basis.to(device), initial_img.flatten())
    initial_img.requires_grad = True

    transposed_basis = basis.T
    P = torch.matmul(binary_pattern, transposed_basis)

    return P, initial_img, output_y, transposed_basis


def main():
    IMG_PATH = 'test_img/mr_president.jpg'
    OUTPUT_PATH = 'test_img/test_5.jpg'

    INPUT_IMG = cv2.imread(IMG_PATH, 0)
    IMG_SIZE = INPUT_IMG.shape

    HISTORY_DIR = 'test_img/rec_history'
    SAVE_RECONSTRUCT_HISTORY = True

    PERCENT = 0.3
    NUM_PATTERNS = int(IMG_SIZE[0] * IMG_SIZE[1] * PERCENT)
    EPOCHS = 200
    LR = 0.9

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    P, initial_img, measurement_y, transposed_basis = preprocess(INPUT_IMG, num_patterns=NUM_PATTERNS, device=DEVICE)

    model = SinglePixelCamera(initial_img,
                              loss=torch.nn.MSELoss(),
                              optimizer=torch.optim.AdamW([initial_img], lr=LR),
                              history=SAVE_RECONSTRUCT_HISTORY)

    model.fit(P.to(DEVICE), measurement_y.to(DEVICE), epochs=EPOCHS)
    # print(model.loss_list)

    output_img = model.get_img
    output_img = dct_img_to_pixels(output_img, transposed_basis, img_size=IMG_SIZE)

    cv2.imwrite(OUTPUT_PATH, output_img)

    if SAVE_RECONSTRUCT_HISTORY:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        for i, img in enumerate(model.img_history_list):
            cv2.imwrite(os.path.join(HISTORY_DIR, f'{i}.jpg'), dct_img_to_pixels(img, transposed_basis, IMG_SIZE))


if __name__ == '__main__':
    main()
