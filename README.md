# 总述
请简练地概括项目的主要贡献，使读者可以快速理解并复现你的工作，包括：
- 原始模型的名称及链接
- 优化效果（精度和加速比），简单给出关键的数字即可，在这里不必详细展开
- 在Docker里面代码编译、运行步骤的完整说明
  - 请做到只要逐行运行你给的命令，就能把代码跑起来，比如从docker pull开始

# 原始模型
## 模型简介
我们（摆一摆队）在本次 TRT2022 复赛中选题的原始模型来源于 ICCV 2021 一篇有关 Vision Transformer 的文章: [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808)，CvT 模型的代码链接参见 Microsoft 公开的[官方实现](https://github.com/microsoft/CvT)。
- CvT 模型的提出主要是为了完成视觉领域的相关任务，如图像分类以及一些下游视觉任务。该模型在 ImageNet 数据集上的实验效果如下：
  - CvT 模型在 ImageNet-1k 预训练后的结果：
  
    | Model  | Resolution | Param | GFLOPs | Top-1 |
    |--------|------------|-------|--------|-------|
    | CvT-13 | 224x224    | 20M   | 4.5    | 81.6  |
    | CvT-21 | 224x224    | 32M   | 7.1    | 82.5  |
    | CvT-13 | 384x384    | 20M   | 16.3   | 83.0  |
    | CvT-21 | 384x384    | 32M   | 24.9   | 83.3  |
  - CvT 模型在 ImageNet-22k 预训练后的结果：
  
    | Model   | Resolution | Param | GFLOPs | Top-1 |
    |---------|------------|-------|--------|-------|
    | CvT-13  | 384x384    | 20M   | 16.3   | 83.3  |
    | CvT-21  | 384x384    | 32M   | 24.9   | 84.9  |
    | CvT-W24 | 384x384    | 277M  | 193.2  | 87.6  |
- CvT 模型的整体结构如下图所示：

  ![](figures/pipeline.svg)
  
  借鉴了经典 CNN 模型的多阶段网络结构，CvT 模型也设计为 3 个阶段，每个阶段都包括 1 个 Token Embedding 步骤和多个 Transformer Block。与 ViT 模型不同的是，CvT 模型中的 Token Embedding 是通过卷积来实现的，并加入了额外的 Layer Normalization，这使得每个阶段能逐渐减少 Token 的数量，同时增加 Token 的宽度，以实现类似经典 CNN 设计的空间下采样并增强语义表示能力。在 Transformer 结构中，通过深度可分离卷积来替换传统 Self Attention QKV 的线性投影，以更高效地建模局部空间上下文信息。这种 CNN 和 Transformer 混合的结构，在保持 Transformer 特性（动态注意力、全局上下文信息、更好泛化能力）的同时，也带来了 CNN 的优点（平移缩放不变性），实验结果也表明了其性能超过了 SOTA CNN 和 Vision Transformer 模型。
  

## 模型优化的难点
如果模型可以容易地跑在TensorRT上而且性能很好，就没有必要选它作为参赛题目并在这里长篇大论了。相信你选择了某个模型作为参赛题目必然有选择它的理由。  
请介绍一下在模型在导出时、或用polygraphy/trtexec解析时、或在TensorRT运行时，会遇到什么问题。换句话说，针对这个模型，我们为什么需要额外的工程手段。

