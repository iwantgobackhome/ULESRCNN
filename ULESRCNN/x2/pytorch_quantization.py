import torch
from PIL import Image
from torchvision import transforms

model = torch.load('./lesrcnn_x2.pt')

# backend = "qnnpack"
backend = "x86"
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
model_static_quantized = torch.quantization.prepare(model, inplace=False)
model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)

# torch.save(model_static_quantized, './lesrcnn_x2_int8.pt')