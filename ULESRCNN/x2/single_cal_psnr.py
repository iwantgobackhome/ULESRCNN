from solver import rgb2ycbcr, psnr
from PIL import Image
import numpy as np

sr = Image.open("C:/Users/Home/Desktop/128to512_ori.jpg").convert("RGB")
sr = np.array(sr).astype(np.float32)

hr = Image.open("C:/Users/Home/Desktop/512.png").convert("RGB")
hr = np.array(hr).astype(np.float32)

hr = rgb2ycbcr(hr) #tcw201904122350
sr = rgb2ycbcr(sr)  
bnd = 4     # 스케일 사이즈
im1 = hr[bnd:-bnd, bnd:-bnd]
im2 = sr[bnd:-bnd, bnd:-bnd]

psnr = psnr(im1, im2)
print(psnr)