#!/bin/sh

# clear

# rm ./*.pb ./*.onnx ./*.plan ./result-*.txt

# trtexec \
#     --onnx=CvT-13-Modify.onnx \
#     --minShapes=input:1x3x224x224\
#     --optShapes=input:16x3x224x224 \
#     --maxShapes=input:32x3x224x224 \
#     --memPoolSize=workspace:10240 \
#     --saveEngine=CvT-13.plan \
#     --shapes=input:16x3x224x224 \
#     --verbose \
#     > result-CvT-13-FP32.txt

polygraphy run CvT-13-Modify.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --save-engine=CvT-13.plan \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes input:[1,3,224,224] \
    --trt-opt-shapes input:[16,3,224,224] \
    --trt-max-shapes input:[32,3,224,224] \
    --input-shapes   input:[16,3,224,224] \
    > result-CvT-13-FP32.txt