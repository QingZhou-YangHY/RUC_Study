### 1.环境准备
基础环境装好了，剩下的缺啥补啥

### 2.模型和数据
- 模型：从 HuggingFace 下载Qwen/Qwen2.5-7B-Instruct 和 Qwen/Qwen2.5-32B-Instruct。
- 任务数据：Maze 任务、或你要用的 RLHF 类似任务。Maze 数据就是 (prompt, answer)，然后写一个 verifier 去检查路径是否正确。

### 3.配置 VERL 训练脚本
在 configs/dapo_qwen.yaml 写类似下面的配置（简化版）：

### 4.启动训练
用 torchrun 或 accelerate 启动分布式训练

### 5.评估
训练完成后，可以直接用 VERL 的评估模块：
VERL 会算出 Pass@1 和 Pass@k。

#### 小tips:
- 7B 模型：单卡 A800 基本能跑，8bit/16bit 都行。
- 32B 模型：建议多卡并行（最好 4~8 张 A800），用 tensor_parallel 或者 fsdp。
- 混合精度：建议开启 --bf16，A800 对 bf16 支持很好。

---

### For Now!

现在在研究的是如果对问题进行拆解，让模型强化学习的效率更快。如果你感兴趣的话，可以熟悉一下迷宫任务，然后RL算法有没有什么可以改进的，包括rollout过程和奖励设计（这两块应该会简单一些）。
这两块可能是我要做的

奖励的定义可以看一下verifier的实现：https://github.com/RUCAIBox/Passk_Training/blob/main/code/maze_verifier.py

---
### 自己的一些idea
之前说想做定理证明那块
可以先看一下有什么训练数据和评测数据，然后跑一跑baseline

---
### 闲来无事？
可以follow dapo那个工作的数据https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k，模型的话用个1.5b的应该就行了