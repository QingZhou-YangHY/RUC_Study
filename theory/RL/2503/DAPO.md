### DAPO: An Open-Source LLM Reinforcement Learning System at Scale  

### DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）梳理
[字节、清华团队开源RL算法DAPO，性能超越DeepSeek的GRPO](https://zhuanlan.zhihu.com/p/31085938827) 👈大致了解DAPO

DAPO的成功得益于其四大核心技术：Clip-Higher、动态采样、Token-Level策略梯度损失和过长奖励整形。
1.Clip Higher，促进系统的多样性，避免熵崩溃；  
2.动态采样，提高训练效率和稳定性；  
3.令牌级策略梯度损失，这在长CoT RL场景中至关重要；  
4.超长奖励整形，减少奖励噪音，稳定训练。  


#### 增加模型探索空间
- DAPO算法移除了KL散度项，从而允许模型在训练过程中自由探索。
- Clip-Higher调整上下裁剪范围，提升低概率token的探索能力，有效缓解熵崩溃问题。

#### 细粒度方面
- 引入动态采样策略，过滤准确率为0或1的无效样本，确保批次内梯度有效性
- 采用Token级策略梯度损失，强化长序列中每个token的贡献，避免低质量模式（如重复生成）的干扰。
    ##### 这块组里好像有相关文章，可以看看

DAPO确实没有什么新意，都是正常人能想到的优化方法