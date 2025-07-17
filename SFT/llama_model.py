# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Llama 为什么是 Decoder-Only？Decoder-Only (如 GPT、Llama)

Llama 模型主要设计用于文本生成任务，特别是续写、问答和对话。在这些任务中，模型需要根据已有的上下文（包括用户提示和它自己已经生成的部分）来预测下一个词。
这种自回归的生成方式天然就适合仅解码器架构。它不需要一个独立的编码器来“理解”一个完整的输入序列并转换成另一种形式，而是通过自身的因果注意力机制来逐步构建对上下文的理解并生成后续内容。
"""
from typing import Callable, Optional, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import check_model_inputs
from .configuration_llama import LlamaConfig


logger = logging.get_logger(__name__)
"""
补充一下__name__笔记

__name__ 是 Python 中的一个内置特殊变量。

当一个模块被直接运行时，__name__ 的值是 "__main__"。

当一个模块被作为其他模块导入时，__name__ 的值就是模块的名称（也就是文件名，不带 .py 后缀）。
"""

@use_kernel_forward_from_hub("RMSNorm")
# 装饰器 (Decorator) 的语法糖.去加载一个专门针对 RMSNorm 操作优化过的核函数,这个核函数可能是一个通用的、高性能的 RMSNorm 实现。然后，装饰器将这个通用的高性能核函数“注入”到 LlamaRMSNorm 类的 forward 方法中。


# Llama 模型使用的层归一化
class LlamaRMSNorm(nn.Module):
    """

    归一化: 不同评价指标往往具有不同的量纲和量纲单位，这样的情况会影响到数据分析的结果.
    为了消除指标之间的量纲影响，需要进行数据标准化处理，其中，最典型的就是数据的归一化处理。

    传统的 LayerNorm 通过减去均值并除以标准差来进行归一化。
    而 RMSNorm 则只通过激活值的均方根 (RMS) 进行归一化，这意味着它不通过减去均值来使激活值居中

    RMSNorm 只关注特征向量的幅度（magnitude）而非其中心点。
    原因:
    1.RMSNorm 认为，规范激活值的幅度对于稳定训练就足够了。
    2.通过不减去均值，RMSNorm 在某种程度上保留了激活值的均值信息。在某些情况下，模型的均值可能包含有用的信息，或者至少不应该被强制归零。

    主要通过计算输入状态的方差来归一化，然后乘以一个可学习的 weight 参数。

    variance_epsilon 用于数值稳定性，防止除以零。
    """
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps # 平滑项,增加到方差中，以防止方差为零时除以零的错误

    def forward(self, hidden_states):
        # 临时转换数据类型,使用更高的精度，到时候会换回去
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # 每个元素平方，沿着最后一个维度（即特征维度 hidden_size）计算均值
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 仿射变换
        # variance是平方的均值，不是方差.这就是 RMSNorm 与 LayerNorm 的核心区别
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # rsqrt更快

        # 补充知识点:
        # 吞吐量（Throughput）是衡量系统处理能力的关键指标，在计算机科学和深度学习领域通常指：
        # 单位时间内系统能完成的任务量或数据处理量。

        # 这一步加上variance_epsilon 以增加数值稳定性，其实就是防止分母为零
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        # 当打印模块对象时，它会提供一个额外的字符串表示.这对于调试和检查模型结构非常有用。
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

# Llama 模型的旋转位置嵌入
class LlamaRotaryEmbedding(nn.Module):
    """
    实现了 旋转位置嵌入 (RoPE)，这是 Llama 模型用于注入位置信息的方式

    它不使用传统的绝对位置嵌入，而是通过旋转矩阵的方式将位置信息融入到 Attention 机制的 Q 和 K 向量中
    
    尽量理解一下吧

    """
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        # rope_init_fn 会根据 config.rope_type 初始化 inv_freq 和 attention_scaling
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    # @dynamic_rope_update 装饰器表明它支持高级的 RoPE 类型，例如动态 RoPE
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)

    # RoPE使用了欧拉公式
    def forward(self, x, position_ids):
        # inv_freq决定每个注意力头维度的旋转基频
        # 低频维度（靠近向量头部）旋转缓慢，捕获长程依赖。
        # 高频维度（靠近向量尾部）旋转快速，捕获局部特征。

        # 扩展 inv_freq 到 [batch_size, num_heads, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)

        # 扩展 position_ids 到 [batch_size, 1, seq_len]
        # position_ids表示输入序列中每个token的绝对位置
        position_ids_expanded = position_ids[:, None, :].float()


        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # 重复频率以匹配向量维度（复数实部+虚部）
            emb = torch.cat((freqs, freqs), dim=-1)
            # 计算旋转矩阵的 cos 和 sin 分量
            # attention_scaling是 RoPE（旋转位置编码）中用于动态调整注意力范围的可选缩放因子，主要解决长文本外推（extrapolation）问题。
            # 通过缩放旋转角度，控制位置编码对相对距离的敏感度。
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# 辅助函数
def rotate_half(x):
    """
    将输入张量的最后维度分成两半并进行旋转,实现复数平面上的90度旋转
    """
    
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
    # 复数视角下对应代码中的 [-x2, x1]

# 辅助函数
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    旋转位置嵌入应用到查询 q 和键 k 张量上，实现 RoPE 的数学计算
    """
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # unsqueeze_dim 指定了 cos 和 sin 应该在哪一个维度上被 unsqueeze（即增加一个大小为 1 的新维度），以便它们能正确地与 q 和 k 进行逐元素相乘
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

#  Llama 模型中的多层感知机（FFN）
class LlamaMLP(nn.Module):
    """
    实现了 Llama 模型中的前馈网络 (FFN)，也称为多层感知机 (MLP)

    它包含三个线性层 (gate_proj, up_proj, down_proj) 和一个激活函数 (act_fn)

    Llama 使用的是 SwiGLU 类型的激活函数结构：down_proj(act_fn(gate_proj(x)) * up_proj(x))
    
    TODO:
    为什么这块要用多层感知机？不是很懂，查阅一下，上面的激活函数原理查阅一下，不懂

    之前的理解有误，“MLP 不好”通常是指仅由 MLP 构成的浅层网络在处理复杂任务（如图像识别、自然语言处理）时的局限性，它们难以捕捉长距离依赖和结构化信息。
    但在 Transformer 这种结合了自注意力机制和 MLP 的复合架构中，MLP 的作用是独特且高效的。
    它不再是孤立工作的，而是作为注意力机制的“后处理器”。



    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        # 给模型提供更宽广的空间来学习和提取特征
        self.intermediate_size = config.intermediate_size 
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """

        SwiGLU 结构
        self.gate_proj(x): 这代表了 SwiGLU 结构中的“门控”分支
        self.up_proj(x): 这代表了 SwiGLU 结构中的“信息”分支
        self.act_fn(...): 在 Llama 模型中，这个激活函数通常是 SiLU (Sigmoid Linear Unit)。SiLU(x)=x⋅σ(x).σ(x) 是 Sigmoid 函数
        逐元素乘法实现“门控”self.act_fn(self.gate_proj(x)) 的每个元素充当一个动态权重或开关，来“门控”或调整 self.up_proj(x) 相应元素的重要性。
        self.down_proj(...): 这个层将特征从 intermediate_size 维度线性变换回原始的 hidden_size 维度。

        """
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

# 辅助函数
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    解决 分组查询注意力 (Grouped Query Attention, GQA) 中的一个问题：查询（Q）头的数量和键（K）、值（V）头的数量不一致。
    把少数的 K/V 头“复制”或者“广播”多次，让它们能够“服务”多个 Q 头。    
    这里用的是Grouped Query Attention (GQA).多查询注意力 (MQA) 是 GQA 的一个特例，它只使用一个 K 头和一个 V 头来服务所有 Q 头。
    和 KV Cache不一样
    """
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # 最难理解的就是这句：hidden_states = hidden_states[:, :, None, :, :].expand(...)
    # 核心复制操作：利用 unsqueeze 和 expand 实现广播
    # 比如 hidden_states 形状是 (B, 2, S, D)
    # hidden_states[:, :, None, :, :] 会变成 (B, 2, 1, S, D)
    # 然后 expand 到 (B, 2, 4, S, D) -> 逻辑上是 [[K1],[K2]] -> [[K1,K1,K1,K1],[K2,K2,K2,K2]]
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # 最终得到一个形状是 (批次, Q_头数, 序列长度, 头维度) 的 K/V 张量，它现在与 Q 张量的头部数量完全匹配，可以进行正常的注意力计算了。
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# 辅助函数
def eager_attention_forward(
    
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    """
    这是一个默认的注意力前向传播函数，用于在没有特定优化（如 Flash Attention）时执行注意力计算。
    它包含了 QKV 矩阵乘法、注意力 Masking、Softmax 和 Dropout
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    #  计算注意力权重
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    # Dropout 以 p 的概率随机将一些注意力权重设置为 0，以防止过拟合。
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    # 将注意力权重与值向量进行加权求和
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

# Llama 模型的多头自注意力机制
class LlamaAttention(nn.Module):
    """
    包含 q_proj, k_proj, v_proj, o_proj 等线性投影层

    集成了 KV Cache (past_key_value) 以加速生成过程

    通过 attention_interface 动态选择注意力实现（默认为 eager_attention_forward，但支持其他优化如 Flash Attention）

    """
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV Cache
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        # 如果配置中指定了非 "eager" 的注意力实现 (如 "flash_attention_2" 或 "sdpa")，
        # 则从预定义的函数字典 ALL_ATTENTION_FUNCTIONS 中获取对应的优化函数
        # 这块是eager所以就不执行
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,  # 训练时应用 Dropout，推理时禁用
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

# Llama 模型的一个 Transformer 解码器层
class LlamaDecoderLayer(GradientCheckpointingLayer):
    """
    继承自 GradientCheckpointingLayer，表明它支持梯度检查点，这是一种内存优化技术，可以在训练大型模型时减少内存使用

    包含一个 self_attn (LlamaAttention) 和一个 mlp (LlamaMLP)

    在每个子层之前都有 RMS Normalization (input_layernorm 和 post_attention_layernorm)，这是 Llama 架构的一个特点

    TODO:
    这块每个子层之前都有 RMS Normalization的作用？

    归一化的核心作用是将这些激活值缩放到一个更稳定的范围（例如，保持方差大致为 1），从而为后续的层提供更一致、更健康的输入分布。
    这有助于平滑损失函数的曲面，使梯度更加稳定和可预测，从而加速收敛。
    
    TODO:
    在这块残差结构的意义？

    如果一个子层的操作是 F(x)，那么残差连接的输出是 x+F(x)
    缓解梯度消失问题,允许梯度直接流回网络的早期层，确保所有层都能接收到有意义的梯度信号，从而使深层网络能够有效训练。

    """
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        # 对 hidden_states 应用 RMS 归一化。这是 Llama 特有的预归一化 (Pre-Normalization) 设计
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        # 保存 MLP 子层的输入，用于第二次残差连接
        residual = hidden_states
        # 在 MLP 进入之前，对注意力子层输出（已经加上了第一个残差）应用 RMS 归一化
        hidden_states = self.post_attention_layernorm(hidden_states)
        # 归一化后的 hidden_states 经过 mlp (LlamaMLP) 模块，该模块应用非线性变换
        hidden_states = self.mlp(hidden_states)
        # MLP 的输出通过第二次残差连接加回到 MLP 的输入上
        hidden_states = residual + hidden_states
        return hidden_states

# Llama 系列模型的基础抽象类
@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    """
    这是所有 Llama 模型变体的基类，继承自 Hugging Face 的 PreTrainedModel
    
    TODO:看看这个PreTrainedModel是什么

    定义了模型的通用配置 (config: LlamaConfig)、
    权重初始化方法 (_init_weights)、对梯度检查点、Flash Attention、SDPA (Scaled Dot Product Attention) 等特性的支持

    _no_split_modules 指定了哪些模块在分布式训练中不应该被分割（例如，为了梯度检查点）
    """
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # _no_split_modules 指定了哪些模块在分布式训练中不应该被分割（例如，为了梯度检查点）
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _supports_static_cache = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LlamaRMSNorm):
            module.weight.data.fill_(1.0)

# 核心的 Llama Transformer 模型，不带任何头部（head）
@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    """
    Llama 模型的主干。它不包含任何任务特定的输出头

    包含词嵌入层 (embed_tokens)、多个 LlamaDecoderLayer (self.layers) 和最终的 RMS Normalization 层 (self.norm)

    集成了 LlamaRotaryEmbedding

    
    """
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 这会创建 config.num_hidden_layers 个（例如，不同 Llama 变体可能有 32、40、80 层）独立的 LlamaDecoderLayer 实例，每个实例都有自己的一套注意力层和 MLP 权重。
        # 每个 LlamaDecoderLayer 都会依次处理 hidden_states。
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 这是最终的 RMS 归一化层.它对最后一个 Transformer 层的输出进行归一化，然后可能将其传递给特定任务的头部
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 初始化旋转位置嵌入 (RoPE) 模块
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        # 一个控制是否启用梯度检查点的标志
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        # 用于执行最终的初始化步骤，例如权重初始化和可能的剪枝
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @check_model_inputs
    @auto_docstring
    # forward 方法定义了数据流，包括嵌入、位置编码、解码器层的迭代以及缓存管理
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # 如果未提供 inputs_embeds，则使用 self.embed_tokens(input_ids) 执行 token 嵌入查找
        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        # 第一次调用模型生成或从头开始一个新序列
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # 未提供 cache_position
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # 未提供 position_ids
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # create_causal_mask:辅助函数，用于生成因果注意力掩码
        # 解码器（如 Llama）中，每个 token 在计算注意力时只能“看到”它自己和它之前的 token，不能“看到”未来的 token。
        # 因果掩码通过将未来 token 的注意力分数设置为一个非常小的负数（例如 -inf），使其在 Softmax 后变为零，从而强制执行这种因果关系。
        # 除了因果掩码，这个函数还会考虑传入的 attention_mask（通常用于处理填充 token），并结合 KV 缓存的位置信息，构建一个最终的组合注意力掩码
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 通过将位置嵌入与隐藏状态相加来获得最终的输入表示
        hidden_states = hidden_states + position_embeddings

        # 依次通过模型的所有 num_hidden_layers 个 LlamaDecoderLayer
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

"""
针对不同任务的模型变体

这些类都继承自 LlamaPreTrainedModel 并封装了 LlamaModel 作为其主干，
然后在其之上添加了针对特定任务的输出层（head）
"""
# 用于因果语言建模（文本生成）的 Llama 模型
@auto_docstring
# LlamaForCausalLM 集成了 Hugging Face Transformers 库的 GenerationMixin
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    """
    用于因果语言建模（例如，文本生成、补全）

    在 LlamaModel 的输出之上添加了一个线性层 lm_head，用于将隐藏状态映射到词汇表大小的 logits

    实现了 GenerationMixin，使其能够使用 Hugging Face Transformers 的生成方法（如 generate）

    forward 方法计算 logits 并可选地计算因果语言建模损失
    """
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"} #  lm_head 的权重在张量并行中按列复制，或采取类似的分布方式
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])} # 定义了流水线并行的输入 (hidden_states) 和输出 (logits)，指明了这个模块在流水线中的位置。

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        """""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """
)


# 用于序列分类任务的 Llama 模型
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    """
    用于序列分类任务（例如，情感分析、文本分类）

    在 LlamaModel 的输出之上添加了一个线性层 score，将隐藏状态映射到类别数量的 logits

    它通常使用最后一个非填充 token 的隐藏状态进行分类
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        # 分类头部
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            # 如果 pad_token_id 未定义，并且批次大小为 1，则 last_non_pad_token 简单地设置为 -1，表示取序列的最后一个 token
            last_non_pad_token = -1
        elif input_ids is not None:
            # 找到每个序列中最右边那个非填充 token 的索引
            # 这是通过创建一个 non_pad_mask（非填充位置为 1，填充位置为 0），然后将其与 token 的原始索引相乘，最后取最大值（argmax）来实现的。
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            # 如果 input_ids 不可用（即传入的是 inputs_embeds），且 pad_token_id 未定义，它会发出警告，并默认取最后一个 token，因为在这种情况下无法识别填充。
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

# 用于问答任务的 Llama 模型
@auto_docstring
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    """
    用于问答任务（例如，抽取式问答）

    在 LlamaModel 的输出之上添加了一个线性层 qa_outputs，预测答案开始和结束位置的 logits
    """
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        # 只是一个命名约定，其功能与 LlamaModel 相同。
        self.transformer = LlamaModel(config)
        """
        问答头部 一个线性层，接收 transformer 的输出 hidden_states（维度为 config.hidden_size），并将其投影到 2 个输出维度
        这两个输出维度分别代表了答案开始位置的 logits 和答案结束位置的 logits
        对于序列中的每个 token，模型都会预测它作为答案开始或结束位置的“可能性”分数。
        """
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> QuestionAnsweringModelOutput:
        outputs: BaseModelOutputWithPast = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state

        # logits 张量的最后一个维度为 2
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # 调用模型的内部损失函数（通常是交叉熵损失）。它会分别计算 start_logits 与 start_positions 之间的损失，以及 end_logits 与 end_positions 之间的损失，并将它们相加作为总损失。
        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# 用于 Token 分类任务的 Llama 模型
@auto_docstring
class LlamaForTokenClassification(LlamaPreTrainedModel):
    """
    用于 Token 分类任务（例如，命名实体识别 NER）

    在 LlamaModel 的每个 token 的隐藏状态之后添加一个 dropout 层和一个线性层 score，用于预测每个 token 的类别
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        # 确定要使用的 Dropout 比率
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        # 初始化 Dropout 层，它将在分类头之前应用于隐藏状态
        self.dropout = nn.Dropout(classifier_dropout)
        # Token 分类头部
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        # 获取 Llama 模型输出的最终隐藏状态
        sequence_output = outputs.last_hidden_state
        # 防止模型在训练时对特定特征过于依赖，从而提高泛化能力
        sequence_output = self.dropout(sequence_output)
        # 递给分类头部 self.score 这个线性层为序列中的每个位置（每个 token） 输出一个 num_labels 维的向量，其中包含了该 token 属于每个类别的预测分数
        logits = self.score(sequence_output)

        loss = None
        # 如果提供了 labels（真实类别标签），则计算损失
        # 感觉只有这块是SFT
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]

"""
总结:

    这份 Llama 模型的代码从底层的归一化、位置嵌入和注意力机制开始构建，逐步构建起单个 Transformer 层，然后是完整的 Llama 模型主干，最后再针对不同的下游任务添加特定的输出层。

    这种分层的设计使得代码易于理解、维护和扩展，同时也充分利用了 Hugging Face Transformers 库的通用基础设施（如 PreTrainedModel、Cache 等）。


"""