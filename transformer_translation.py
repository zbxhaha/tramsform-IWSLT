# transformer_translation.py
"""
完整的Encoder-Decoder Transformer实现
专门针对机器翻译任务优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """多头注意力机制 - 翻译专用版本"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性变换矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        # 线性变换并分头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # softmax和dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        return self.W_o(attn_output), attn_weights


class PositionWiseFFN(nn.Module):
    """位置前馈网络"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class LayerNorm(nn.Module):
    """层归一化"""

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(self_attn_output)
        x = self.norm1(x)

        # 交叉注意力
        cross_attn_output, _ = self.cross_attn(x, memory, memory, src_mask)
        x = x + self.dropout(cross_attn_output)
        x = self.norm2(x)

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)

        return x


class TransformerForTranslation(nn.Module):
    """用于机器翻译的完整Transformer模型"""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 512, num_heads: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 d_ff: int = 2048, max_seq_length: int = 5000,
                 dropout: float = 0.1, positional_encoding: str = 'sinusoidal',
                 share_embeddings: bool = False):
        super().__init__()

        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.share_embeddings = share_embeddings

        # 词嵌入
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        if share_embeddings and src_vocab_size == tgt_vocab_size:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码
        if positional_encoding == 'sinusoidal':
            self.src_pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_length)
            self.tgt_pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_length)
        elif positional_encoding == 'none':
            # 使用恒等位置编码
            self.src_pos_encoding = IdentityPositionalEncoding(d_model, max_seq_length)
            self.tgt_pos_encoding = IdentityPositionalEncoding(d_model, max_seq_length)
        else:
            raise ValueError(f"不支持的位置编码: {positional_encoding}")

        self.dropout = nn.Dropout(dropout)

        # 编码器
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # 解码器
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

        # 如果共享词嵌入，将输出层权重与目标词嵌入绑定
        if share_embeddings and src_vocab_size == tgt_vocab_size:
            self.output_layer.weight = self.tgt_embedding.weight

        # 参数初始化
        self._init_parameters()

    def _init_parameters(self):
        """参数初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """创建padding mask"""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def create_look_ahead_mask(self, seq_len: int) -> torch.Tensor:
        """创建look ahead mask（用于解码器）"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码器前向传播"""
        # 源序列嵌入 + 位置编码
        src_embedded = self.dropout(self.src_pos_encoding(self.src_embedding(src)))

        # 编码器层
        memory = src_embedded
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)

        return memory

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               src_mask: Optional[torch.Tensor] = None,
               tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """解码器前向传播"""
        # 目标序列嵌入 + 位置编码
        tgt_embedded = self.dropout(self.tgt_pos_encoding(self.tgt_embedding(tgt)))

        # 解码器层
        output = tgt_embedded
        for layer in self.decoder_layers:
            output = layer(output, memory, src_mask, tgt_mask)

        return output

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_pad_idx: int = 0, tgt_pad_idx: int = 0) -> torch.Tensor:
        """前向传播 - 训练时使用teacher forcing"""

        # 创建mask
        src_mask = self.create_padding_mask(src, src_pad_idx)
        tgt_mask = self.create_padding_mask(tgt, tgt_pad_idx) & \
                   self.create_look_ahead_mask(tgt.size(1)).to(src.device)

        # 编码器-解码器
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, src_mask, tgt_mask)

        return self.output_layer(output)

    def generate(self, src: torch.Tensor, max_len: int = 50,
                 src_pad_idx: int = 0, tgt_bos_idx: int = 2,
                 tgt_eos_idx: int = 3) -> torch.Tensor:
        """生成翻译结果 - 推理时使用"""
        self.eval()

        with torch.no_grad():
            # 编码源序列
            src_mask = self.create_padding_mask(src, src_pad_idx)
            memory = self.encode(src, src_mask)

            # 初始化目标序列（开始标记）
            batch_size = src.size(0)
            tgt = torch.ones(batch_size, 1).fill_(tgt_bos_idx).long().to(src.device)

            for _ in range(max_len - 1):
                tgt_mask = self.create_padding_mask(tgt, tgt_eos_idx) & \
                           self.create_look_ahead_mask(tgt.size(1)).to(src.device)

                output = self.decode(tgt, memory, src_mask, tgt_mask)
                logits = self.output_layer(output[:, -1, :])
                next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
                tgt = torch.cat([tgt, next_token], dim=1)

                # 如果所有序列都生成了结束标记，则停止
                if (next_token == tgt_eos_idx).all():
                    break

            return tgt

    def count_parameters(self) -> int:
        """统计参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TranslationModelWithBeamSearch(TransformerForTranslation):
    """带束搜索的翻译模型"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def beam_search(self, src: torch.Tensor, beam_size: int = 5, max_len: int = 50,
                    src_pad_idx: int = 0, tgt_bos_idx: int = 2, tgt_eos_idx: int = 3):
        """束搜索生成"""
        self.eval()

        with torch.no_grad():
            batch_size = src.size(0)

            # 编码源序列
            src_mask = self.create_padding_mask(src, src_pad_idx)
            memory = self.encode(src, src_mask)

            # 初始化束
            beams = [{
                'sequence': torch.tensor([[tgt_bos_idx]], device=src.device),
                'score': 0.0,
                'finished': False
            }]

            for step in range(max_len):
                new_beams = []

                for beam in beams:
                    if beam['finished']:
                        new_beams.append(beam)
                        continue

                    # 获取当前序列
                    tgt = beam['sequence']

                    # 创建mask
                    tgt_mask = self.create_padding_mask(tgt, tgt_eos_idx) & \
                               self.create_look_ahead_mask(tgt.size(1)).to(src.device)

                    # 解码
                    output = self.decode(tgt, memory.expand(tgt.size(0), -1, -1),
                                         src_mask.expand(tgt.size(0), -1, -1, -1), tgt_mask)
                    logits = self.output_layer(output[:, -1, :])
                    probs = F.log_softmax(logits, dim=-1)

                    # 获取top-k候选
                    topk_probs, topk_tokens = torch.topk(probs, beam_size, dim=-1)

                    for i in range(beam_size):
                        new_seq = torch.cat([tgt, topk_tokens[:, i:i + 1]], dim=1)
                        new_score = beam['score'] + topk_probs[0, i].item()
                        finished = (topk_tokens[0, i] == tgt_eos_idx).item()

                        new_beams.append({
                            'sequence': new_seq,
                            'score': new_score,
                            'finished': finished
                        })

                # 选择得分最高的beam_size个序列
                new_beams.sort(key=lambda x: x['score'], reverse=True)
                beams = new_beams[:beam_size]

                # 如果所有序列都结束了，提前停止
                if all(beam['finished'] for beam in beams):
                    break

            # 返回最佳序列
            best_beam = max(beams, key=lambda x: x['score'])
            return best_beam['sequence']


def create_translation_model(src_vocab_size, tgt_vocab_size, config):
    """创建翻译模型"""
    return TransformerForTranslation(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        d_ff=config.d_ff,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout,
        positional_encoding=config.positional_encoding,
        share_embeddings=config.share_embeddings
    )


"""
完整的Encoder-Decoder Transformer实现
专门针对机器翻译任务优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """多头注意力机制 - 翻译专用版本"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性变换矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        # 线性变换并分头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # softmax和dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        return self.W_o(attn_output), attn_weights


class PositionWiseFFN(nn.Module):
    """位置前馈网络"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class LayerNorm(nn.Module):
    """层归一化"""

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class IdentityPositionalEncoding(nn.Module):
    """恒等位置编码 - 什么都不做，用于消融实验"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # 这个类什么都不做，只是为了接口一致

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x  # 直接返回输入，不添加位置信息


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(self_attn_output)
        x = self.norm1(x)

        # 交叉注意力
        cross_attn_output, _ = self.cross_attn(x, memory, memory, src_mask)
        x = x + self.dropout(cross_attn_output)
        x = self.norm2(x)

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)

        return x


class TransformerForTranslation(nn.Module):
    """用于机器翻译的完整Transformer模型"""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 512, num_heads: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 d_ff: int = 2048, max_seq_length: int = 5000,
                 dropout: float = 0.1, positional_encoding: str = 'sinusoidal',
                 share_embeddings: bool = False):
        super().__init__()

        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.share_embeddings = share_embeddings

        # 词嵌入
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        if share_embeddings and src_vocab_size == tgt_vocab_size:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码 - 修复：支持 'none' 选项
        if positional_encoding == 'sinusoidal':
            self.src_pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_length)
            self.tgt_pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_length)
        elif positional_encoding == 'none':
            # 使用恒等位置编码
            self.src_pos_encoding = IdentityPositionalEncoding(d_model, max_seq_length)
            self.tgt_pos_encoding = IdentityPositionalEncoding(d_model, max_seq_length)
        else:
            raise ValueError(f"不支持的位置编码: {positional_encoding}")

        self.dropout = nn.Dropout(dropout)

        # 编码器
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # 解码器
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

        # 如果共享词嵌入，将输出层权重与目标词嵌入绑定
        if share_embeddings and src_vocab_size == tgt_vocab_size:
            self.output_layer.weight = self.tgt_embedding.weight

        # 参数初始化
        self._init_parameters()

    def _init_parameters(self):
        """参数初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """创建padding mask"""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def create_look_ahead_mask(self, seq_len: int) -> torch.Tensor:
        """创建look ahead mask（用于解码器）"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码器前向传播"""
        # 源序列嵌入 + 位置编码
        src_embedded = self.dropout(self.src_pos_encoding(self.src_embedding(src)))

        # 编码器层
        memory = src_embedded
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)

        return memory

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               src_mask: Optional[torch.Tensor] = None,
               tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """解码器前向传播"""
        # 目标序列嵌入 + 位置编码
        tgt_embedded = self.dropout(self.tgt_pos_encoding(self.tgt_embedding(tgt)))

        # 解码器层
        output = tgt_embedded
        for layer in self.decoder_layers:
            output = layer(output, memory, src_mask, tgt_mask)

        return output

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_pad_idx: int = 0, tgt_pad_idx: int = 0) -> torch.Tensor:
        """前向传播 - 训练时使用teacher forcing"""

        # 创建mask
        src_mask = self.create_padding_mask(src, src_pad_idx)
        tgt_mask = self.create_padding_mask(tgt, tgt_pad_idx) & \
                   self.create_look_ahead_mask(tgt.size(1)).to(src.device)

        # 编码器-解码器
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, src_mask, tgt_mask)

        return self.output_layer(output)

    def generate(self, src: torch.Tensor, max_len: int = 50,
                 src_pad_idx: int = 0, tgt_bos_idx: int = 2,
                 tgt_eos_idx: int = 3, temperature: float = 0.8) -> torch.Tensor:
        """生成翻译结果 - 推理时使用"""
        self.eval()

        print(f"[DEBUG] 生成开始 - src shape: {src.shape}")
        print(f"[DEBUG] 特殊标记 - bos: {tgt_bos_idx}, eos: {tgt_eos_idx}, pad: {src_pad_idx}")

        with torch.no_grad():
            # 编码源序列
            src_mask = self.create_padding_mask(src, src_pad_idx)
            memory = self.encode(src, src_mask)
            print(f"[DEBUG] 编码完成 - memory shape: {memory.shape}")

            # 初始化目标序列（开始标记）
            batch_size = src.size(0)
            tgt = torch.ones(batch_size, 1).fill_(tgt_bos_idx).long().to(src.device)
            print(f"[DEBUG] 初始tgt: {tgt}")

            for step in range(max_len - 1):
                print(f"[DEBUG] 步骤 {step} - 当前tgt: {tgt}")

                # 创建mask - 修复可能的mask问题
                tgt_padding_mask = self.create_padding_mask(tgt, tgt_eos_idx)
                look_ahead_mask = self.create_look_ahead_mask(tgt.size(1)).to(src.device)
                tgt_mask = tgt_padding_mask & look_ahead_mask.unsqueeze(0).unsqueeze(0)

                print(f"[DEBUG] tgt_mask shape: {tgt_mask.shape}")

                output = self.decode(tgt, memory, src_mask, tgt_mask)
                logits = self.output_layer(output[:, -1, :])

                print(f"[DEBUG] 步骤 {step} - logits shape: {logits.shape}")
                print(f"[DEBUG] 步骤 {step} - logits范围: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

                # 使用温度采样而不是argmax，避免重复
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                print(f"[DEBUG] 步骤 {step} - 下一个token: {next_token}")

                tgt = torch.cat([tgt, next_token], dim=1)

                # 检查是否所有序列都生成了结束标记
                eos_generated = (next_token == tgt_eos_idx)
                print(f"[DEBUG] 步骤 {step} - EOS生成: {eos_generated}")

                if eos_generated.all():
                    print(f"[DEBUG] 所有序列生成结束标记，提前停止")
                    break

            print(f"[DEBUG] 最终生成结果: {tgt}")
            return tgt

    def count_parameters(self) -> int:
        """统计参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TranslationModelWithBeamSearch(TransformerForTranslation):
    """带束搜索的翻译模型"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def beam_search(self, src: torch.Tensor, beam_size: int = 5, max_len: int = 50,
                    src_pad_idx: int = 0, tgt_bos_idx: int = 2, tgt_eos_idx: int = 3):
        """束搜索生成"""
        self.eval()

        with torch.no_grad():
            batch_size = src.size(0)

            # 编码源序列
            src_mask = self.create_padding_mask(src, src_pad_idx)
            memory = self.encode(src, src_mask)

            # 初始化束
            beams = [{
                'sequence': torch.tensor([[tgt_bos_idx]], device=src.device),
                'score': 0.0,
                'finished': False
            }]

            for step in range(max_len):
                new_beams = []

                for beam in beams:
                    if beam['finished']:
                        new_beams.append(beam)
                        continue

                    # 获取当前序列
                    tgt = beam['sequence']

                    # 创建mask
                    tgt_mask = self.create_padding_mask(tgt, tgt_eos_idx) & \
                               self.create_look_ahead_mask(tgt.size(1)).to(src.device)

                    # 解码
                    output = self.decode(tgt, memory.expand(tgt.size(0), -1, -1),
                                         src_mask.expand(tgt.size(0), -1, -1, -1), tgt_mask)
                    logits = self.output_layer(output[:, -1, :])
                    probs = F.log_softmax(logits, dim=-1)

                    # 获取top-k候选
                    topk_probs, topk_tokens = torch.topk(probs, beam_size, dim=-1)

                    for i in range(beam_size):
                        new_seq = torch.cat([tgt, topk_tokens[:, i:i + 1]], dim=1)
                        new_score = beam['score'] + topk_probs[0, i].item()
                        finished = (topk_tokens[0, i] == tgt_eos_idx).item()

                        new_beams.append({
                            'sequence': new_seq,
                            'score': new_score,
                            'finished': finished
                        })

                # 选择得分最高的beam_size个序列
                new_beams.sort(key=lambda x: x['score'], reverse=True)
                beams = new_beams[:beam_size]

                # 如果所有序列都结束了，提前停止
                if all(beam['finished'] for beam in beams):
                    break

            # 返回最佳序列
            best_beam = max(beams, key=lambda x: x['score'])
            return best_beam['sequence']


def create_translation_model(src_vocab_size, tgt_vocab_size, config):
    """创建翻译模型"""
    print(f"[DEBUG] 创建模型参数:")
    print(f"  - src_vocab_size: {src_vocab_size}")
    print(f"  - tgt_vocab_size: {tgt_vocab_size}")
    print(f"  - positional_encoding: {config.positional_encoding}")
    print(f"  - d_model: {config.d_model}")

    try:
        model = TransformerForTranslation(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            d_ff=config.d_ff,
            max_seq_length=config.max_seq_length,
            dropout=config.dropout,
            positional_encoding=config.positional_encoding,
            share_embeddings=config.share_embeddings
        )
        print(f"[DEBUG] 模型创建成功!")
        return model
    except Exception as e:
        print(f"[DEBUG] 模型创建失败: {e}")
        raise