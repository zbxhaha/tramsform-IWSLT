# Transformer从零实现 - 大模型课程作业

## 项目简介
本项目完整实现了Transformer架构，支持Encoder-only语言建模和Encoder-Decoder机器翻译任务。

## 主要特性
- ✅ Multi-Head Self-Attention
- ✅ Position-wise FFN  
- ✅ 残差连接 + LayerNorm
- ✅ 正弦位置编码
- ✅ 完整的Encoder-Decoder架构
- ✅ 束搜索解码
- ✅ 消融实验分析
- ✅ 训练可视化

## 快速开始

### 环境设置
```bash
# 创建环境
conda create -n transformer python=3.10
conda activate transformer

# 安装依赖
pip install -r requirements.txt