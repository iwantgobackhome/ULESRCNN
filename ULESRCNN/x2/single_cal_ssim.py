from solver import calculate_ssim, rgb2ycbcr
from PIL import Image
import numpy as np

sr = Image.open("C:/Users/Home/Desktop/128to256_ori.jpg").convert("RGB")
sr = np.array(sr).astype(np.float32)

hr = Image.open("C:/Users/Home/Desktop/256.png").convert("RGB")
hr = np.array(hr).astype(np.float32)

hr = rgb2ycbcr(hr) #tcw201904122350
sr = rgb2ycbcr(sr)

bnd = 2   # 스케일 사이즈
im1 = hr[bnd:-bnd, bnd:-bnd]
im2 = sr[bnd:-bnd, bnd:-bnd]

print(calculate_ssim(im1, im2))
