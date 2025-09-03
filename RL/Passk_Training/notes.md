### Pass@k Training for Adptively Balancing Eplortion and Exploitation of LRMs

[Pass@k Training for Adptively Balancing Eplortion and Exploitation of LRMs](https://github.com/RUCAIBox/Passk_Training)👈仓库链接

Pass@k 训练系统解决了 RLVR 训练中的一个根本性限制：传统的 Pass@1 奖励机制会导致策略偏向保守操作，最终收敛于局部最优。核心创新在于用 Pass@k 指标取代了 Pass@1 指标，从而更好地平衡了探索和利用。

#### 传统 Pass@1 问题
在传统的 RLVR 训练中，策略会根据单个响应是否正确获得二元奖励。这会产生一种保守的偏差，即模型会避免探索，以最大限度地降低生成错误响应的风险。

#### Pass@k 解决方案
Pass@k 方法为每个问题生成 k 个答案，并使用其中最大的奖励作为训练信号。这鼓励探索，同时保持开发效益。

