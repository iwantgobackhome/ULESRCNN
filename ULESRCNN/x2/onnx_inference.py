import onnxruntime as ort
import numpy as np
from PIL import Image
import time
import torchvision.transforms as transforms

# ONNX 런타임 세션 생성
session = ort.InferenceSession("lesrcnn_x2.onnx")

# 이미지 처리 및 ONNX 입력 준비
image = Image.open('C:/Users/Home/Desktop/128.png').convert("RGB")
tf = transforms.ToTensor()
input_tensor = tf(image).unsqueeze(0).numpy()  # 이미지를 NumPy 배열로 변환

# ONNX 모델을 사용하여 추론 실행
inputs = {session.get_inputs()[0].name: input_tensor}
t1 = time.time()
outputs = session.run(None, inputs)
t2 = time.time()
output_tensor = outputs[0]

# 결과 출력
print("Output shape:", output_tensor.shape)
print("Inference time:", t2 - t1, "seconds")

# 이미지 저장 및 표시
output_image = output_tensor.squeeze(0).transpose(1, 2, 0)
output_image = (output_image * 255).clip(0, 255).astype(np.uint8)
im = Image.fromarray(output_image)
# im.show()
im.save("C:/Users/Home/Desktop/128to256_com.png")
