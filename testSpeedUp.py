import torch
import numpy as np
import os
import time
import sys
import onnxruntime
import tensorrt as trt
from cuda import cudart


sys.path.append("./lib")
torch.manual_seed(66)

# 加载 .pt 模型并测试其在 PyTorch 上的推理耗时 -------------------------------------------
model = torch.load('CvT-13.pt')
model.eval()

input_data = torch.randn(1, 3, 224, 224, dtype=torch.float32, device='cuda')

nRound = 50
with torch.no_grad():
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(nRound):
        model(input_data)
    torch.cuda.synchronize()
    time_pytorch = (time.time() - t0) / nRound
print('PyTorch time:', time_pytorch)


# 加载 .onnx 模型并测试其在 onnxruntime 上的推理耗时 ----------------------------------------
ort_session = onnxruntime.InferenceSession("./CvT-13-Modify.onnx", providers=["CUDAExecutionProvider"])
ort_inputs = {ort_session.get_inputs()[0].name:input_data.cpu().numpy()}

torch.cuda.synchronize()
t0 = time.time()
for i in range(nRound):
    ort_outs = ort_session.run(None, ort_inputs)
torch.cuda.synchronize()
time_ort = (time.time() - t0) / nRound
print('onnxruntime time:', time_ort)


# 加载 .plan 模型并测试其在 TensorRT 上的推理耗时 -------------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
if os.path.isfile("./CvT-13.plan"):
    with open("./CvT-13.plan", 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine == None:
        print("Failed loading engine!")
        exit()
    print("Succeeded loading engine!")
else:
    print("Failed finding engine!")
    exit()

context = engine.create_execution_context()
context.set_binding_shape(0, [1, 3, 224, 224])
_, stream = cudart.cudaStreamCreate()

data = input_data.cpu().numpy()
inputH0 = np.ascontiguousarray(data.reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
_, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

torch.cuda.synchronize()
t0 = time.time()
for i in range(nRound):
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
torch.cuda.synchronize()
time_trt = (time.time() - t0) / nRound
print('TensorRT time:', time_trt)

cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
cudart.cudaStreamSynchronize(stream)

print('Speedup1:', time_pytorch / time_trt)
print('Speedup2:', time_ort / time_trt)

cudart.cudaStreamDestroy(stream)
cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)




