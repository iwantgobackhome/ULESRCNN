import importlib
import torch
from collections import OrderedDict
from PIL import Image
from torchvision import transforms
import time
import numpy as np

'''
# pth 로드
module = importlib.import_module("model.{}".format("lesrcnn"))
net = module.Net(scale=3, 
                     group=1)
state_dict = torch.load("./checkpoint/lesrcnn_x3_573000.pth")
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k
    # name = k[7:] # remove "module."
    new_state_dict[name] = v

net.load_state_dict(new_state_dict)
'''
# pt모델 로드
net = torch.load('./lesrcnn_x3.pt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device = "cpu"  # cpu 고정

net = net.to(device)

image = Image.open("C:/Users/Home/Desktop/4.png").convert("RGB")
# image.resize((256, 256))
# image.show()

tf = transforms.ToTensor()
lr = tf(image)

t1 = time.time()
lr = lr.unsqueeze(0).to(device)
sr = net(lr, 2).detach().squeeze(0)
lr = lr.squeeze(0)
t2 = time.time()

'''
# psnr 계산
from solver import rgb2ycbcr, psnr
hr = Image.open("C:/Users/Home/Desktop/768.png").convert("RGB")
hr = np.array(hr).astype(np.float32)
sr = sr.cpu()
sr = sr.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
hr = rgb2ycbcr(hr) #tcw201904122350
sr = rgb2ycbcr(sr)  
bnd = 2
im1 = hr[bnd:-bnd, bnd:-bnd]
im2 = sr[bnd:-bnd, bnd:-bnd]

psnr = psnr(im1, im2)
print(psnr)

print(sr.shape)
print(t2 - t1)
'''


# 이미지 저장
sr = sr.cpu()
ndarr = sr.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
im = Image.fromarray(ndarr)
im.show()
im.save("C:/Users/Home/Desktop/171to513_com.png")
