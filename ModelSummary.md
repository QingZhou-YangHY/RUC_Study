## GPT2:无监督预训练和自回归生成
Language Models are Unsupervised Multitask Learners
和Bert一样,只不过Bert是双向，GPT2是单向
**比当时同期的模型所用的数据多+无监督学习**:GPT-2 使用了超过 40GB 的文本数据，训练了一个拥有 15 亿个参数的模型，这远超当时的其他模型。这使得它能够学习到更加复杂和细致的语言特征。  
无监督学习的优势：通过在大量文本数据上进行训练，GPT-2 能够进行零-shot学习，即即使没有在特定任务上进行专门的训练，它也能在新任务中展示出惊人的性能。  
**自回归生成模型**:每生成一个 token（词或子词），它都会基于之前生成的文本（上下文）来预测下一个 token。这种生成方式使得 GPT-2 在生成长篇连贯、语法正确且富有逻辑性的文本时，表现出了前所未有的能力。  

## Llama3:
The Llama 3 Herd of Models
纯自回归语言建模,无监督预训练
训练数据的量上来了15T Token,分组查询注意力(GQA)，和Llama2一样.包括后面的归一化	RMSNorm,激活函数SwiGLU.参数量达 8B/70B（Llama 3 70B 是当前开源最强模型之一）

## DeepSeek-R1:
Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
纯语言模型，多模态依托
无监督预训练
MoE（Mixture of Experts，混合专家）架构
总参数量：145B，但 激活参数量仅 20B/Token
主要是他做强化学习的过程，通过强化学习来提升模型的推理能力

## COT
#### Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
结论:思维链只能对100B+的模型有影响
思考:能不能刚开始训练的时候用较长的，随着学习进行不断缩短，Gemini的思考速度似乎是最快的那一档.像人一样熟能生巧(data argument).成为teacher再变成小模型student
<br>
Gemini读ip极其严重(讨厌这一点)  
Cursor,Claude这些收费的暂时不说，code也不用

## DeepSeek-V3是MoE架构(保留意见)  
和Diffusion Model一个问题,如果现在(2025)年让我选择，我可能去选择轻量化，蒸馏模型.(部署)  
以后要是出息了，手上有资源再钻研这块吧

vllm: https://zhuanlan.zhihu.com/p/694138714

PHD:

如果要做的话，极有可能:复杂推理相关任务，具体来说就是让大模型解决数学题和其他各学科的竞赛题  

监督微调：JiuZhang 3.0: Efficiently Improving Mathematical Reasoning by Training Small Data Synthesis Models
强化学习：Improving large language models via fine-grained reinforcement learning with minimum editing constraint
Agent：ChatCoT: Tool-Augmented Chain-of-Thought Reasoning on Chat-based Large Language Models