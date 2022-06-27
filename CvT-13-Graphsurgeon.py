import onnx
import onnx_graphsurgeon as gs
import numpy as np


# 读取 CvT-13.onnx 并进行调整
graph = gs.import_onnx(onnx.load("CvT-13.onnx"))

##################################### 修改1 ###############################################
# 找到 Squeeze_5724 节点
Squeeze_5724 = [node for node in graph.nodes if node.name == "Squeeze_5724"][0]

# 指定 Squeeze 操作要进行压缩的维度
Squeeze_5724.attrs["axes"] = np.array([1])

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "CvT-13-Modify.onnx")

print("finish CvT-13 onnx-graphsurgeon!")
