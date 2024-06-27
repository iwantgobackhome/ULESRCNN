import torch.onnx
import torch
from onnx import shape_inference, numpy_helper, version_converter
import onnx
import onnxruntime as ort
from onnxruntime.tools.convert_onnx_models_to_ort import convert_onnx_models_to_ort, OptimizationStyle
import numpy as np
from PIL import Image
from torchvision import transforms
import time
from onnxruntime.quantization import quantize_dynamic, QuantType, shape_inference, quantize_static, CalibrationDataReader
from pathlib import Path

#Function to Convert to ONNX 
def Convert_ONNX(model): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, 3, 1024, 1024, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         (dummy_input,2),       # model input (or a tuple for multiple inputs) 
         "./lesrcnn_x2.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size', 2: 'height', 3: 'width'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size', 2: 'height', 3: 'width'}}) 
    print(" ") 
    print('Model has been converted to ONNX')
  
# psnr 계산용 전처리도 해줌    
def calcul_psnr(hr_src, model_output):
    from solver import rgb2ycbcr, psnr
    hr = Image.open(hr_src).convert("RGB")
    hr = np.array(hr).astype(np.float32)
    sr = model_output
    hr = rgb2ycbcr(hr) #tcw201904122350
    sr = rgb2ycbcr(sr)  
    bnd = 2
    im1 = hr[bnd:-bnd, bnd:-bnd]
    im2 = sr[bnd:-bnd, bnd:-bnd]

    return psnr(im1, im2)
    
    
### onnx 모델로 전환
# net = torch.load('./lesrcnn_x2.pt')
# net.eval()
# Convert_ONNX(net)
# onnx.save(onnx.shape_inference.infer_shapes(onnx.load('./lesrcnn_x2.onnx')), './lesrcnn_x2.onnx')

### onnx 모델 체크
# onnx_model = onnx.load('./lesrcnn_x2.onnx')
# onnx.checker.check_model(onnx_model)

# inferred_model = shape_inference.infer_shapes(onnx_model)
# print(inferred_model.graph.value_info[0])


### onnx 모델 양자화

# 정적 양자화
class QuntizationDataReader(CalibrationDataReader):
    def __init__(self, data_loader, input_name):

        self.torch_dl = data_loader

        self.input_name = input_name
        self.datasize = len(self.torch_dl)

        self.enum_data = iter(self.torch_dl)

    def to_numpy(self, pt_tensor):
        return pt_tensor.detach().cpu().numpy() if pt_tensor.requires_grad else pt_tensor.cpu().numpy()

    def get_next(self):
        batch = next(self.enum_data, None)
        if batch is not None:
          return {self.input_name: self.to_numpy(batch[0])}
        else:
          return None

    def rewind(self):
        self.enum_data = iter(self.torch_dl)
        

'''
model_prep_path = './lesrcnn_x2_pre.onnx'
# shape_inference.quant_pre_process('./lesrcnn_x2.onnx', model_prep_path, skip_symbolic_shape=False)

from dataset import TrainDataset, TestDataset
from torch.utils.data import DataLoader

test_data = TestDataset('../dataset/Urban100', scale=2)
test_loader = DataLoader(test_data, batch_size=1, num_workers=0, shuffle=False)

qdr = QuntizationDataReader(test_loader, input_name='modelInput')

q_static_opts = { "ActivationSymmetric" : False , 
                 "WeightSymmetric" : True } 
if torch.cuda.is_available(): 
  q_static_opts = { "ActivationSymmetric" : True , 
                  "WeightSymmetric" : True } 

model_int8_path = './lesrcnn_x2_int8.onnx'
quantized_model = quantize_static(model_input=model_prep_path,
                                               model_output=model_int8_path,
                                               calibration_data_reader=qdr,
                                               extra_options=q_static_opts)
'''

# 추론

opt = ort.SessionOptions()
opt.graph_optimization_level= ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
opt.log_severity_level=3
opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

ort_session = ort.InferenceSession('./lesrcnn_x2.onnx', opt)

# 이미지 전처리
image = Image.open("C:/Users/Home/Desktop/128.png").convert("RGB")
lr = np.array(image).astype(np.float32)
lr = lr/255
lr = np.transpose(lr, (2, 0, 1))
lr = np.expand_dims(lr, axis=0)

# 이미지 추론
t1 = time.time()
output = ort_session.run(None, {'modelInput': lr})[0]
t2 = time.time()

# 추론된 데이티 처리
output = np.squeeze(output)
output = np.transpose(output, (1, 2, 0))
output = (output * 255).clip(0, 255)
output = output.astype(np.uint8)
i = Image.fromarray(output)
# i.show()
i.save("C:/Users/Home/Desktop/128to256_com.png")

# psnr 계산
psnr = calcul_psnr("C:/Users/Home/Desktop/256.png", output)
print(psnr)
print(t2 - t1)


# onnx모델을 ort모델로 변환
# convert_onnx_models_to_ort(model_path_or_dir=Path('./lesrcnn_x2_6.onnx'), optimization_styles=[OptimizationStyle.Fixed],target_platform='arm')