#!/usr/bin/env python3
"""
Transformer机器翻译实验启动脚本 - 支持IWSLT2017官方数据集
"""

import argparse
import os
import sys

# 添加当前目录到路径，确保可以导入本地模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from translation_trainer import TranslationExperiment, TranslationConfig
    from translation_data_manager import print_translation_dataset_info
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有必要的模块文件都在当前目录")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Transformer机器翻译实验 - IWSLT2017')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='iwslt2017',
                        choices=['iwslt2017', 'ted_talks'],
                        help='翻译数据集名称')
    parser.add_argument('--language-pair', type=str, default='en-de',
                        choices=['en-de', 'de-en', 'en-fr', 'fr-en', 'en-it', 'it-en'],
                        help='语言对')
    parser.add_argument('--use-huggingface', action='store_true',
                        help='使用Hugging Face官方数据集（默认使用本地数据）')
    parser.add_argument('--data-dir', type=str, default='./en-de',
                        help='本地数据目录路径（当不使用Hugging Face时）')
    parser.add_argument('--max-length', type=int, default=100,
                        help='最大序列长度')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批大小')

    # 模型参数
    parser.add_argument('--d-model', type=int, default=512,
                        help='模型维度')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='注意力头数')
    parser.add_argument('--num-encoder-layers', type=int, default=6,
                        help='编码器层数')
    parser.add_argument('--num-decoder-layers', type=int, default=6,
                        help='解码器层数')
    parser.add_argument('--d-ff', type=int, default=2048,
                        help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout率')
    parser.add_argument('--positional-encoding', type=str, default='sinusoidal',
                        choices=['sinusoidal', 'none'],
                        help='位置编码类型')
    parser.add_argument('--share-embeddings', action='store_true',
                        help='共享源语言和目标语言词嵌入')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='梯度裁剪值')

    # 实验参数
    parser.add_argument('--ablation', action='store_true',
                        help='运行消融实验')
    parser.add_argument('--save-dir', type=str, default='translation_results',
                        help='结果保存目录')
    parser.add_argument('--cache-dir', type=str, default='./data_cache',
                        help='数据集缓存目录（当使用Hugging Face时）')

    args = parser.parse_args()

    # 显示数据集信息
    print_translation_dataset_info()

    # 设置配置
    experiment = TranslationExperiment()

    # 更新配置
    experiment.config.dataset_name = args.dataset
    experiment.config.language_pair = args.language_pair
    experiment.config.use_huggingface = args.use_huggingface
    experiment.config.max_seq_length = args.max_length
    experiment.config.batch_size = args.batch_size
    experiment.config.d_model = args.d_model
    experiment.config.num_heads = args.num_heads
    experiment.config.num_encoder_layers = args.num_encoder_layers
    experiment.config.num_decoder_layers = args.num_decoder_layers
    experiment.config.d_ff = args.d_ff
    experiment.config.dropout = args.dropout
    experiment.config.positional_encoding = args.positional_encoding
    experiment.config.share_embeddings = args.share_embeddings
    experiment.config.num_epochs = args.epochs
    experiment.config.learning_rate = args.learning_rate
    experiment.config.weight_decay = args.weight_decay
    experiment.config.grad_clip = args.grad_clip
    experiment.config.save_dir = args.save_dir
    experiment.config.cache_dir = args.cache_dir

    # 设置数据目录
    experiment.config.data_dir = args.data_dir

    print(f"\n实验配置:")
    print(f"- 数据集: {args.dataset}")
    print(f"- 语言对: {args.language_pair}")
    print(f"- 数据源: {'Hugging Face' if args.use_huggingface else '本地数据'}")
    if not args.use_huggingface:
        print(f"- 数据目录: {args.data_dir}")
    print(f"- 模型维度: {args.d_model}")
    print(f"- 编码器层数: {args.num_encoder_layers}")
    print(f"- 解码器层数: {args.num_decoder_layers}")
    print(f"- 训练轮数: {args.epochs}")
    print(f"- 批大小: {args.batch_size}")
    print()

    # 运行实验
    if args.ablation:
        print("运行完整翻译实验（包含消融实验）...")
        experiment.run_full_translation_experiment()
    else:
        print("运行基线翻译实验...")
        experiment.run_baseline_experiment()


if __name__ == '__main__':
    main()