from camera import Camera
import cv2
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

pattern_num = 2000
img = cv2.imread('lena64.jpeg', 0)
img = torch.FloatTensor(img) / 255
size_img = img.shape

model = Camera(size_img=size_img, pattern_num=pattern_num)

basis = model.get_dct_matrix()
pattern = model.get_binary_mask()
y = model.detector(pattern, img.flatten())

img = torch.randint(0, 2, (size_img[0], size_img[1]), dtype=torch.float32, device=device)


img = torch.matmul(basis.to(device), img.flatten())
basis = basis.T
p = torch.matmul(pattern, basis)

img.requires_grad = True
model.compile(img, loss=torch.nn.MSELoss, optimizer=torch.optim.AdamW, lr=0.03)
model.fit(p.to(device), y.to(device), epochs=1000)

img = model.get_img()

cv2.imwrite('test.jpg',
            torch.matmul(basis, img.to('cpu')).reshape(size_img).numpy() * 255)
