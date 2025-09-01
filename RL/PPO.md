### Proximal Policy Optimization Algorithms
相对来说比较简单的一篇文章

### motivation + 回顾TRPO论文的一些结论和推理过程
TRPO的提出主要是解决deep Q-learning的一些缺点(训练非常不稳定,训练的时间非常长但是performance不能得到保证)
TRPO把优化引入了深度强化学习当中.对于当前执行的policyΠ1通过采样生成一个优化问题optimization1.通过解optimization保证一定能够生成更好的Π2,performance是有提升的.如此不断地循环往复.


理解TRPO对于理解PPO是至关重要的
TRPO是一种Policy Gradient的方法.
$\nabla$θ = $\hat{g}$ =第t步的积分没看懂的公式
由于存在自动求导，上面的式子等价于一个目标.也是一个公式
这样做理论上没问题，但是现实中会遇到问题:下一个θ'和原来的θ的策略差的非常多(有问题,他说和原来比非常差,但是怎么知道刚开始做的performance就是好的?那不是先入为主的错误?)
因此提出了TRPO方法,提出了一种方法:把新老policy的KL divergency作为约束项(让这两个policy不要差的太多),同时maximize优化目标.在TRPO这篇论文里也提到了一些NConstrain的一种方法,将KL divergency作为优化目标的一部分,原来的TRPO训练时间非常长，转换到NConstrain，整个的训练时间就会非常快.
PPO本质上就是这个思路的一个具体化,PPO这篇论文提出了两个方法.

### approach重点
一种方法是section3的Clipped Surrogate Objective,另一种方法是section4的Adaptive KL Penalty.
这两种方法在最后比较之后Clipped Surrogate Objective是比较好的.

Clipped Surrogate Objective是怎么工作的?
上式要优化的目标记为$L^{CPI}(θ)$ = 两个policy的ratio乘advantage-function  在t步的积分.同时Clipped Surrogate Objective可以写成下面的式子(比较复杂)

Adaptive KL Penalty:
这个方法比较简洁,初始的提出在TRPO文章里面.

---

上面做的笔记有些晦涩，所以重新梳理了一遍

## 梳理

[无需RL基础理解 PPO 和 GRPO](https://zhuanlan.zhihu.com/p/27704969958)👈看这个可以补补PPO，GRPO基础


#### PPO 依赖于 Actor + Critic + 裁剪 + KL 惩罚框架。
**引入Critic**：使用“预测分数线”来改进奖励
在 RL 中，这个“分数线”被称为价值函数。它充当一个基准。我们的训练目标从仅奖励转变为超过该基准的程度，由优势函数表示。

**添加裁剪和最小值操作**：防止过度更新。在PPO（近端策略优化）中，这种平衡通过“裁剪”机制实现。通俗点说就是设置上限下限。

**参考模型**：防止作弊和极端策略。在大语言模型领域，类似的情况是生成有害或虚假内容以人为提高某些奖励指标。  


**GRPO：用“多次模拟平均值”替代价值函数。** 详见GRPO

PPO-clip(用的比较多)  
前面求期望的部分包含了state,old probs,action,reward,next state.  
**价值value(critic)**:输入一个状态，输出一个价值  
一共有3个网络:old policy,new policy(actor),value(critic)  
**优势函数（Advantage Function）** 用来衡量在某个状态下，采取某个特定动作比“平均表现”好多少。  

如何应用到Large Language Model?  

**因果语言模型Causal Language Model** :后面的字能看到前面的，但是前面的看不到后面的(**自回归**)  

词表logit那里分类  

如何训练一个奖励模型?  
1.收集问题，让大语言模型输出结果。人类进行排序(排序比打分容易)  
2.训练一个人类偏好的奖励模型(reward model),用来给大语言模型的输出打分,用Bradley-Terry(主要用于体育赛事)算法计算loss    
3.Train policy with PPO  

错位现象  
具体的还是看图解吧  
