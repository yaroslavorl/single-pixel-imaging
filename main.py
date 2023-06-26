from camera import Camera
import cv2
import torch

pattern_num = 2000
img = cv2.imread('/home/jaroslav/PycharmProjects/dcan/single_pixel_camera/img/lena64.jpeg', 0)
img = torch.FloatTensor(img) / 255
size_img = img.shape

model = Camera(size_img=size_img, pattern_num=pattern_num)

basis = model.get_dct_matrix()
pattern = model.binary_mask()
y = model.detector(pattern, img.flatten())

img = torch.randint(0, 2, (size_img[0], size_img[1]), dtype=torch.float32)

img = torch.matmul(basis, img.flatten())
basis = basis.T
p = torch.matmul(pattern, basis)

img.requires_grad = True
model.compile(img, loss=torch.nn.MSELoss, optimizer=torch.optim.AdamW, lr=0.03)
model.fit(p, y, epochs=800)

img = model.get_img()

cv2.imwrite('/home/jaroslav/PycharmProjects/dcan/single_pixel_camera/img/test.jpg',
            torch.matmul(basis, img).reshape(size_img).numpy() * 255)
