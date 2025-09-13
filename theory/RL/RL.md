# Deep Reinforcement Learning(RL)
Introduction of Deep Reinforcement Learning
Supervised Learning -> RL
比如下围棋，如何去找一个最好的答案.收集答案困难 + 人们无法标注 : RL
## RL和ML一样都是三个步骤
Machine Learning = Looking for a Function
Actor Environment进行互动 Environment 对 Actor(function)进行Observation(input) Actor进行Action(output).
## Actor就是我们要找的function       Action = f(Observation)
Environment 对 Actor Reward. Find a policy maximizing total reward
![RLProcess](..\images\basic_theroy\RLProcess.png "RLProcess")
Example: Playing Video Game ( Space invader )   Learning to play Go 
Machine Learning
Step 1: function with unknown 
Step 2: define loss from training data
Step 3: optimization

## RL
## Step 1: function with unknown    function:Policy Network(Actor) 输入是a vector or a pixels 输出是每一个行为
![step1](..\images\basic_theroy\step1.png "step1")
实际上这就是Classification Task!!!  图片? 也许CNN.作业中是FC.  
如果考虑全部以前是RNN,现在是transformer
采取Sample based on scores.用概率随机是很好的，而不是看谁softmax后最大取谁.
## Step 2: Define "Loss" 
![step2_1](..\images\basic_theroy\step2_1.png "step2_1")
![step2_2](..\images\basic_theroy\step2_2.png "step2_2")
每一行为得到的叫reward,Total reward(一个episode的总和)叫return.reward是某一个行为的时候得到的,把整个episode的加起来叫return.return(total reward)是我们要最大化的目标.
把-Total reward当成Loss 越小越好.
## Step 3: Optimization
![step3](..\images\basic_theroy\step3.png "step3")    
Trajectory t = {s1,a1,s2,a2,...}trajectory翻译过来叫轨迹比较好一点? 
Reward (function)要看s1和a1得到r1,有的游戏可以只看a1,但这么做通常来讲不对
Reward是一个function   
Optimization要找到一个Network参数放在Actor里面可以让R(t)越大越好.
a1是sample产生的.这不是一个普通的Network,具有随机性.Env 和 Reward不是Network,只是一个黑盒子.Reward是一条规则.Reward(有些)和Env都具有随机性.

RL最主要的问题是 optimization. c.f.GAN   GAN又干了!   只不过这里面的Reward,Env不是Network.所以没有办法用Gradient Descent.


# Policy Gradient (重要)
控制actor的方法:
![control_actor_1](..\images\basic_theroy\control_actor_1.png "control_actor_1")
让机器学习看到xx情况下xx本质上就是分类问题:把预期结果定义为label,然后计算输出和label的Cross-entropy,让他最小.如果让他不干xx,就取个负号即可. 
![control_actor_2](..\images\basic_theroy\control_actor_2.png "control_actor_2")
还可以计算L = e1 - e2. 很像train一个classifier,控制Actor的行为.
![control_actor_3](..\images\basic_theroy\control_actor_3.png "control_actor_3")

a和$\hat{a}$ 的Cross-entropy在后面简记为e
每一个行为的好坏程度简记为$A_i$,$A_i$的正负代表是否期望他发生，绝对值代表希望程度
==猜测:在探索过程中,这里是不是可以把$A_i$变成一个超参数来进行动态学习?==
![control_actor_4](..\images\basic_theroy\control_actor_4.png "control_actor_4")


怎么定义A?
## Version 0(不正确的版本)
![version_0](..\images\basic_theroy\version_0.png "version_0")
收集训练资料Training Data,通常是many episodes为了收集到足够多的资料.每一对{$s_i$,$a_i$}是{observation,action}.$A_i$ = $r_i$
很明显这是一个Short-sighted Version,因为每一个行为不是独立的,每一个行为action都会影响到接下来发生的事情.而且会有Reward delay:有时候需要牺牲现在的reward来获得long-term reward.这个是很符合人类认知的.所以这个version是不正确的,记录在这里的原因是为了学习一下改进这个思路.
但是这个implement RL的时候特别容易犯这个错误!


## Version 1
![version_1](..\images\basic_theroy\version_1.png "version_1")
G1 = r1 + r2 + r3 + ... + rN 来评估a1的好坏    G2 = r2 + r3 + ... + rN 来评估a2的好坏 ......
G:cumulated reward
Version 1的问题在于可能会抢功劳！
但我感觉这个方法的思路起点是好的，这样可以找到是因为前面发生了xx所以才导致了xx，可能是式子表达的不对或者说缺少了一个判断，没有一个东西来忽略哪些不重要哪些重要.


## Version 2 
![version_2](..\images\basic_theroy\version_2.png "version_2")
Discout facctor γ < 1   G1' = r1 + γr2 + γ2r3 + ...  距离越远γ平方越多
A1 = G1' A2 = G2' ......   
到这里已经合理多了,但是感觉还是不可以.他没有真正识别出来，只是表达出如果步数差得多，那么他一定就没关系.显然有很多问题

==不同的RL方法是在A上面下文章，可能我也是很容易这么去想着来改进==

## Version 3
![version_3](..\images\basic_theroy\version_3.png "version_3")
Good or bad reward is "relative"
If all the rn >= 10   rn = 10 is negative...   reward是相对的  
我们需要做标准化 所有的G' 减去b    B叫baseline
如何设定 baseline b ?在接下来的版本会提到

## Policy Gradient是怎么操作的?
![PolicyGradient_1](..\images\basic_theroy\PolicyGradient_1.png "PolicyGradient_1")
Initialize actor network parameters θ0
For training iteration i = 1 to T
 Using actor θi-1 to interact
  Obtain data {s1,a1},{s2,a2},...,{sN,aN}
  Compute A1,A2,..,AN  (这里面一定要改)
  Compute loss L
  θi = θi-1 - η * grad(L)  和gradient descent是一样的
![PolicyGradient_2](..\images\basic_theroy\PolicyGradient_2.png "PolicyGradient_2")
一般的training data collection 都是在training 之外的，但是RL的data collection是在training循环里面的,所以非常费时间
同一个Action对于不同的Actor的作用效果是不一样的,所以上面的资料只能训练θi-1，不能训练θi.因此就需要不断更新data. e.g.棋魂中大马步飞和小马步飞
![PolicyGradient_3](..\images\basic_theroy\PolicyGradient_3.png "PolicyGradient_3") 
![PolicyGradient_4](..\images\basic_theroy\PolicyGradient_4.png "PolicyGradient_4") 
==但我觉得可以改进吧，万一偶然在之前发现了很好的action没有利用上呢?这样就会特别死板，不能挖掘天赋，太循规蹈矩了==
==虽然听起来action很容易乱,但是做成了可能就比较开创性吧==
==或者说可以先根据一个资料用几次然后再更新.这样能让他把经验总结全==

上面的是On-policy:train和interact的Actor是同一个    Off-policy:要训练的Actor 和 与环境互动的Actor是两个.这样就不用在每个epoch收集资料了. 
![PolicyGradient_5](..\images\basic_theroy\PolicyGradient_5.png "PolicyGradient_5") 

经典的Off-policy做法Proximal Policy Optimization(PPO).重点是train的Actor要知道自己和interact的Actor的difference.interact的actor的行为有些可以采纳,有些不行
==一些具体的做法影片希望有时间可以看看学习一下==
![PolicyGradient_6](..\images\basic_theroy\PolicyGradient_6.png "PolicyGradient_6") 


Exploration(训练过程中非常重要的技巧)：data collection里面具有随机性(随机性十分重要). e.g.Enlarge output entropy  Add noises onto parameters.
![PolicyGradient_7](..\images\basic_theroy\PolicyGradient_7.png "PolicyGradient_7") 
==所以我们可以在这里集群学习?就是上面的变形?好多个Actor共同学习，只不过是在前面的Action不同,所以导致后面的结果不一样.但是这样不用担心"因为一开始的选择导致了速度很慢/无法学明白"这样的问题==

DeepMind - PPO (和OpenAI同时提出)应用于一些机器人.

## Critic
Critic: Given actor θ,how good it is when observing s (and taking action a).看到某一个游戏画面，预测将来可能会得到的reward
一种Critic:Value function $V^{θ}(s)$: When using actor θ,the discounted cumulated reward expects to be obtained after seeing s.输入是s,通过$V^{θ}(s)$输出一个scalar(数值),这个scalar是the discounted cumulated reward即Version 2中的式子$G_{i}'$
Value function的数值和观察的actor有关系,根据自己水平(参数θ)进行猜测.对某一个actor来说，看到某一个游戏画面预测接下来得到的the discounted cumulated reward.(未卜先知)

Critic是如何被训练出来的?

- Monte-Carlo(MC) based approach:玩了很多场游戏.看到s就知道$V^{θ}(s)$了.
![MC](..\images\basic_theroy\MC.png "MC") 
- Temporal-differnce(TD) approach:不用玩完正常游戏,只需要上下一点就可以更新Vθ的参数.
![TD](..\images\basic_theroy\TD.png "TD")
MC表示了前后有影响,TD似乎可以减少sample造成的影响
![MCvsTD](..\images\basic_theroy\MCvsTD.png "MCvsTD")  


Critic如何应用在RL?

## Version 3.5
确定了$b$是$V^{θ}(s_{i})$
![version_3.5_1](..\images\basic_theroy\version_3.5_1.png "version_3.5_1")
为什么$b$是$V^{θ}(s_{i})$?
不知道可能会采取什么动作，所以先给一个平均值，也就是$V^{θ}(s_{i})$
![version_3.5_2](..\images\basic_theroy\version_3.5_2.png "version_3.5_2")

上面把随机出来的action的reward当成$G_{t}'$真的好吗?引出了version 4
## Version 4
Advantage Actor-Critic
![version_4](..\images\basic_theroy\version_4.png "version_4")
小tip:可以共用一些参数
![tip_of_actor-critic](..\images\basic_theroy\tip_of_actor-critic.png "tip_of_actor-critic")

## Outlook:Deep Q Network(DQN) 
直接用Critic就决定用什么action
==Rainbow用了七种变形,有许多录音课程可以看==
![Outlook_DQN](..\images\basic_theroy\Outlook_DQN.png "Outlook_DQN")

# RL很吃运气,很看sample的怎么样

## Reward Shaping
==可能这里是我们要研究，做出改变的==

如果我们不知道actions是不是好的(reward基本上都是0),类似于下围棋问题,我们该如何解决?
想办法提供额外的reward引导agent去学习,这个叫reward shaping.
e.g.曹操望梅止渴
![SparseReward](..\images\basic_theroy\SparseReward.png "SparseReward")
用RL玩VizDoom
,当时第一名就用了Reward Shaping.可以自己定义一些加分扣分的规则.需要一些我们对问题的一些理解.感觉要是有些问题让他自己学习这个方法有很大局限性.
Reward Shaping - Curiosity
看到有意义的新东西就会加分.但是要克服无意义的新.出自2017年ICML的文章.

## No Reward:Learning from Demonstration
在一些任务里面才有reward,但是很多任务是没有reward的.比如自动驾驶.
机器会有神逻辑,所以有时候会意想不到发生什么...这里面非常能体现他很蠢...所以提出的解决办法其中之一就是limitation Learning
没有reward,但是有demonstration(通常是人类的).有点像Supervised Learning.这确实是,同时也也叫Behavior Cloning.但是会产生问题:人类和机器观察到的s是不一样的.所以有的地方他是不知道怎么处理了.内涵某自动驾驶(.除此之外,有些个人的特征不需要模仿的他也学习了,只会完全复制老师行为.

Inverse Reinforcement Learning(IRL)
让机器自己定reward
![IRL](..\images\basic_theroy\IRL.png "IRL")
![IRL_theroy](..\images\basic_theroy\IRL_theroy.png "IRL_theroy")
![IRL_picture](..\images\basic_theroy\IRL_picture.png "IRL_picture")
GAN和IRL本质上是同一种思想
![GANvsIRL](..\images\basic_theroy\GANvsIRL.png "GANvsIRL")
IRL常常训练机械手臂
更潮的做法:给他画面让他学习