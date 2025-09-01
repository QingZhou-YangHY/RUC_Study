## 归一化 (Normalization)就是都变成0-1之间的数  
对每一层的输入/输出进行数值标准化，避免梯度爆炸或消失  
常见方法：Batch Normalization (BN)，Layer Normalization (LN)，RMSNorm / Pre-LN 变种  
## 正则化 (Regularization)防止过拟合、提升泛化能力  
在训练过程中对模型的参数或训练方式“加约束”，避免模型把训练数据背死  
常见方法：L1/L2 正则化，Dropout，数据增强 (Data Augmentation)，早停 (Early Stopping)

## on-policy
行为策略与目标策略相同
on-policy里面只有一种策略，它既为目标策略又为行为策略。
## off-policy
将收集数据当做一个单独的任务
数据来源于一个单独的用于探索的策略(不是最终要求的策略)