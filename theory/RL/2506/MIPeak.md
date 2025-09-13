# Demystifying Reasoning Dynamics with Mutual Information: Thinking Tokens are Information Peaks in LLM Reasoning  
强化学习结合推理  
在LRM的推理过程中，特定生成步骤的MI突然显著增加。我们从理论上分析了这种现象，并表明随着MI的增加，模型预测误差的概率降低。此外，这些MI峰值通常对应于表示反射或转换的标记，如“嗯”、“等待”和“因此”，我们称之为思维标记。然后，我们证明这些思维令牌对LRM的推理性能至关重要，而其他令牌的影响最小。在这些分析的基础上，我们提出了两种简单而有效的方法，通过巧妙地利用这些思维标记来提高LRM的推理性能。  
## 现在做的很多都是类似细粒度化的思想  
文章理论性很强，大部分都是证明，没有细看。感觉是那种只要能想出来这个点子就算成功那种。  
