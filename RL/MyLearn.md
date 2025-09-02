### 归一化 (Normalization)就是都变成0-1之间的数  

对每一层的输入/输出进行数值标准化，避免梯度爆炸或消失  
常见方法：Batch Normalization (BN)，Layer Normalization (LN)，RMSNorm / Pre-LN 变种  
### 正则化 (Regularization)防止过拟合、提升泛化能力  

在训练过程中对模型的参数或训练方式“加约束”，避免模型把训练数据背死  
常见方法：L1/L2 正则化，Dropout，数据增强 (Data Augmentation)，早停 (Early Stopping)

---

## On-policy & Off-policy
[强化学习中的奇怪概念(一)——On-policy与off-policy](https://zhuanlan.zhihu.com/p/346433931)👈
## On-policy
##### the target and the behavior polices are the same
行为策略与目标策略相同,SARSA算法即为典型的on-policy的算法
on-policy里面只有一种策略，它既为目标策略又为行为策略。

### SARSA算法

[【强化学习】02. SARSA 算法原理与代码实现](https://zhuanlan.zhihu.com/p/688429797)👈

SARSA（State-Action-Reward-State-Action）是一种基于时序差分（TD）学习的在线强化学习算法。它通过从智能体与环境的交互中**学习一个动作价值函数（Q 函数），用于评估在给定状态下采取特定动作的价值**。
#### 时序差分学习

时序差分学习是一种介于蒙特卡罗方法和动态规划之间的强化学习方法。它结合了蒙特卡罗方法的样本更新和动态规划的自举更新。在时序差分学习中，智能体在每一步更新价值函数时**只需观察当前状态、动作、奖励和下一个状态**。

#### SARSA 更新动作价值函数

$Q_{t+1}(s, a) = Q_{t}(s, a) + \alpha · (R(s,a) + \gamma ·Q_{t}(s', a') - Q_{t}(s,a))$ 

其中:
- $Q(s, a)$ 是当前状态 $s_t$ 和动作 $a_t$ 的动作价值。
- $\alpha$ 是学习率，控制着价值函数更新的速度。
- $R(s, a)$ 是智能体在执行动作 $a$ 后获得的奖励。
- $\gamma$ 是折扣因子，用于衡量未来奖励相对于即时奖励的重要性。
- $Q_t(s', a')$ 是下一个状态 $s'$ 和相应动作 $a'$ 的动作价值。

---

## Off-policy
##### the learning is from the data off the target policy
- 有时候off-policy需要与重要性采样配合使用
> 为了让方差更小，如果很稳定的话当然不需要！

- Q-Learning算法(或DQN)身为off-policy可以不用重要性采样


将收集数据当做一个单独的任务
数据来源于一个单独的用于探索的策略(不是最终要求的策略)


#### 重要性采样（Importance Sampling）
[重要性采样（Importance Sampling）](https://zhuanlan.zhihu.com/p/41217212)👈
##### 重要性采样的核心是variance reduction，而不是不知道原始分布π(x)

##### 蒙特卡洛积分

高数中通过微元，小长方形面积来代替难以解析出来的曲线的面积

##### 重要性采样的具体内容

上述的估计方法随着取样数的增长而越发精确，但我们希望在一定的抽样数量基础上来增加准确度，减少方差。

人为地对抽样的分布进行干预！在原函数对积分贡献大的区域获得更多的采样机会。

此时采样不是均匀分布的，小矩形的“宽”并不等长，所以我们要对其进行加权，这个权重就是重要性权重。

重要性权重 = 目标分布概率 / 采样分布概率

目标分布概率:我们是知道$π(x)$的
采样分布概率:我们刚才人工干预出来的一个概率

我们这么做就是为了让结果的方差更小，重要性权重的计算不是问题，我们是知道原函数的概率分布的。

> 方差大意味着采样有很大的偶然性，对实际应用非常不友好


### Epsilon-Greedy

$ϵ-greedy$

[强化学习中的Epsilon-Greedy算法](https://zhuanlan.zhihu.com/p/644629552)👈
#### Exploration(探索) vs Exploitation(利用)

Exploration 使agent能够提高其当前对每个动作的了解，从而有望带来长期利益。提高估计动作值的准确性，使agent能够在未来做出更明智的决策。

通过Exploitation agent估计当前的动作价值来选择贪婪的动作来获得最大的奖励。

- 确定性策略:一个将状态空间映射到动作空间的函数。它本身没有随机性质，因此通常会结合$ϵ-greedy$或往动作值中加入高斯噪声的方法来增加策略的随机性。
- 随机性策略:条件为$S_{t}∈S$情况下，动作$A_{t}$的条件概率分布。它本身带有随机性，获取动作时只需对概率分布进行采样即可。

> Q 函数:在状态 s 采取动作 a 之后，按最优策略继续走下去，能拿到的平均累计回报

---