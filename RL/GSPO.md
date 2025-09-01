

### GSPO梳理
[详解Qwen3-GSPO和DeepSeek-GRPO两大强化学习算法的区别](https://zhuanlan.zhihu.com/p/1932791271363154917) 👈大致了解GSPO

把奖励计算从token级别 改成了sequence 级别，解决了LLM 过多的关注token导致训练不稳定的问题

#### GRPO 和 GSPO Loss的计算关键差异
其中最主要的差异在重要性计算那一步，GRPO是计算每个token的概率比
```python
log_ratio = per_token_logps - old_per_token_logps  # 每个token的log概率差
log_importance_weights = log_ratio  # 保留token级粒度
coef_1 = torch.exp(log_importance_weights)  # 每个token的概率比
```

而GSPO是计算整个句子的平均概率比

```python 
log_ratio = per_token_logps - old_per_token_logps  # 每个token的log概率差
# 按句子平均：总log概率差 / 有效token数（避免padding影响）
log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
log_importance_weights = log_importance_weights.unsqueeze(-1)  # 扩展为(batch_size, 1)
coef_1 = torch.exp(log_importance_weights)  # 整个句子的平均概率比

```

如果这样看不懂建议去上面知乎链接看一下例子，这里不赘述了。
