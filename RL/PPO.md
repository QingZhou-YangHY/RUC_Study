# Proximal Policy Optimization Algorithms
相对来说比较简单的一篇文章

## motivation + 回顾TRPO论文的一些结论和推理过程
TRPO的提出主要是解决deep Q-learning的一些缺点(训练非常不稳定,训练的时间非常长但是performance不能得到保证)
TRPO把优化引入了深度强化学习当中.对于当前执行的policyΠ1通过采样生成一个优化问题optimization1.通过解optimization保证一定能够生成更好的Π2,performance是有提升的.如此不断地循环往复.


理解TRPO对于理解PPO是至关重要的
TRPO是一种Policy Gradient的方法.
$\nabla$θ = $\hat{g}$ =第t步的积分没看懂的公式
由于存在自动求导，上面的式子等价于一个目标.也是一个公式
这样做理论上没问题，但是现实中会遇到问题:下一个θ'和原来的θ的策略差的非常多(有问题,他说和原来比非常差,但是怎么知道刚开始做的performance就是好的?那不是先入为主的错误?)
因此提出了TRPO方法,提出了一种方法:把新老policy的KL divergency作为约束项(让这两个policy不要差的太多),同时maximize优化目标.在TRPO这篇论文里也提到了一些NConstrain的一种方法,将KL divergency作为优化目标的一部分,原来的TRPO训练时间非常长，转换到NConstrain，整个的训练时间就会非常快.
PPO本质上就是这个思路的一个具体化,PPO这篇论文提出了两个方法.

## approach重点
一种方法是section3的Clipped Surrogate Objective,另一种方法是section4的Adaptive KL Penalty.
这两种方法在最后比较之后Clipped Surrogate Objective是比较好的.

Clipped Surrogate Objective是怎么工作的?
上式要优化的目标记为$L^{CPI}(θ)$ = 两个policy的ratio乘advantage-function  在t步的积分.同时Clipped Surrogate Objective可以写成下面的式子(比较复杂)

Adaptive KL Penalty:
这个方法比较简洁,初始的提出在TRPO文章里面.
这个方法里面的关于新老policy差异给予的奖励机制是不是太过简单了?我觉得这里可以有说法的。

## experiment
没写笔记，感觉不是很重要?
人形机器人(李宏毅老师的课上放过这个踢足球的动画)
## limitation重点
TRPO里面有一个，两个策略非常将近的时候才成立的前提,这块可以改进?