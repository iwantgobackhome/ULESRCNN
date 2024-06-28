# import importlib
import torch
# from collections import OrderedDict
import cv2
from torchvision import transforms
from tqdm import tqdm
# import time
# import numpy as np

# pt모델 로드
net = torch.load('../x3/lesrcnn_x3.pt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device = "cpu"  # cpu 고정

net = net.to(device)
tf = transforms.ToTensor()

cap = cv2.VideoCapture('C:/Users/Home/Desktop/아즈망가대왕1-1.mp4')
size = (1440, 1080)
out = cv2.VideoWriter(filename='C:/Users/Home/Desktop/아즈망가대왕.mp4', fourcc=cv2.VideoWriter_fourcc('m','p','4','v'), fps=24, frameSize=size)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lr = tf(frame)
    lr = lr.unsqueeze(0).to(device)
    sr = net(lr, 2).detach().squeeze(0)
    sr = sr.cpu()
    ndarr = sr.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    frame = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
    
    out.write(frame)
    cv2.imshow('---', frame)
    if cv2.waitKey(1)&0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
out.release()
    
