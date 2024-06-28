import torch
from PIL import Image
from torchvision import transforms
import time

# 모델과 장치 설정
net = torch.load('./lesrcnn_x2.pt', map_location=torch.device('cpu'))  # 모델을 CPU로 직접 로드
device = torch.device("cpu")  # device를 CPU로 설정
net = net.to(device)  # 모델을 CPU로 이동
net.eval()  # 모델을 평가 모드로 설정


# 이미지 로딩 및 변환
image = Image.open('C:/Users/Home/Desktop/256.png').convert("RGB")
tf = transforms.ToTensor()
lr = tf(image).unsqueeze(0).to(device)  # 이미지를 텐서로 변환하고 바로 장치에 올림

# 추론 실행
with torch.no_grad():  # 불필요한 계산 제거
    t1 = time.time()
    sr = net(lr, 2).squeeze(0).detach()  # 이미지를 네트워크에 통과시키고, 바로 디태치
    t2 = time.time()

# 시간 출력 및 이미지 처리
print("Output shape:", sr.shape)
print("Inference time:", t2 - t1, "seconds")

# CPU로 데이터 이동 및 이미지 저장
sr = sr.cpu()
ndarr = sr.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
im = Image.fromarray(ndarr)
im.show()
