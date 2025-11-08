#!/usr/bin/env python3
"""
IWSLT2017数据管理器 - 支持真实本地数据加载和Hugging Face数据
"""

import os
import xml.etree.ElementTree as ET
import re
import torch
from typing import Dict, List, Tuple, Any
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np

# 数据集信息
TRANSLATION_DATASET_INFO = {
    'iwslt2017': {
        'name': 'IWSLT2017 TED Talks',
        'language_pairs': ['en-de', 'de-en', 'en-fr', 'fr-en', 'en-it', 'it-en'],
        'description': 'TED演讲多语言翻译数据集，包含口语化文本',
        'size': '~200k 平行句对 (英语-德语)'
    },
    'ted_talks': {
        'name': 'TED Talks Multilingual',
        'language_pairs': ['en-de', 'de-en', 'en-fr', 'fr-en'],
        'description': 'TED演讲多语言翻译',
        'size': '~150k-200k 平行句对'
    }
}


def parse_iwslt2017_train_file(file_path: str) -> List[str]:
    """解析IWSLT 2017训练文件，提取纯文本内容"""
    sentences = []

    if not os.path.exists(file_path):
        print(f"警告: 文件不存在 {file_path}")
        return sentences

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            # 跳过空行和元数据行（以<开头的行）
            if line and not line.startswith('<'):
                sentences.append(line)
    except Exception as e:
        print(f"读取训练文件错误 {file_path}: {e}")

    return sentences


def parse_iwslt2017_xml_file(file_path: str) -> List[str]:
    """解析IWSLT 2017 XML格式的开发/测试文件"""
    sentences = []

    if not os.path.exists(file_path):
        print(f"警告: 文件不存在 {file_path}")
        return sentences

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # 提取所有seg标签中的文本
        for seg in root.iter('seg'):
            if seg.text and seg.text.strip():
                sentences.append(seg.text.strip())

    except ET.ParseError as e:
        print(f"XML解析错误 {file_path}: {e}")
        # 如果XML解析失败，尝试简单的文本提取
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 提取seg标签内容
                seg_matches = re.findall(r'<seg id="\d+">(.*?)</seg>', content)
                sentences = [match.strip() for match in seg_matches if match.strip()]
        except Exception as e2:
            print(f"备用解析也失败 {file_path}: {e2}")

    return sentences


def load_local_iwslt2017_data(data_dir: str, language_pair: str = "en-de") -> DatasetDict:
    """
    加载本地IWSLT 2017真实数据

    Args:
        data_dir: 数据目录路径
        language_pair: 语言对，如 "en-de"

    Returns:
        DatasetDict对象，包含train、validation、test分割
    """
    src_lang, tgt_lang = language_pair.split('-')

    print(f"加载本地IWSLT 2017 {language_pair} 数据从: {data_dir}")

    # 训练数据
    train_src_file = os.path.join(data_dir, f"train.tags.{language_pair}.{src_lang}")
    train_tgt_file = os.path.join(data_dir, f"train.tags.{language_pair}.{tgt_lang}")

    train_src = parse_iwslt2017_train_file(train_src_file)
    train_tgt = parse_iwslt2017_train_file(train_tgt_file)

    # 确保对齐
    min_train_len = min(len(train_src), len(train_tgt))
    train_src = train_src[:min_train_len]
    train_tgt = train_tgt[:min_train_len]

    print(f"训练数据: {len(train_src)} 个句子对")

    # 开发数据 (dev2010)
    dev_src_file = os.path.join(data_dir, f"IWSLT17.TED.dev2010.{language_pair}.{src_lang}.xml")
    dev_tgt_file = os.path.join(data_dir, f"IWSLT17.TED.dev2010.{language_pair}.{tgt_lang}.xml")

    dev_src = parse_iwslt2017_xml_file(dev_src_file)
    dev_tgt = parse_iwslt2017_xml_file(dev_tgt_file)

    # 确保对齐
    min_dev_len = min(len(dev_src), len(dev_tgt))
    dev_src = dev_src[:min_dev_len]
    dev_tgt = dev_tgt[:min_dev_len]

    print(f"开发数据: {len(dev_src)} 个句子对")

    # 测试数据 (tst2010-tst2015)
    test_years = ['2010', '2011', '2012', '2013', '2014', '2015']
    test_src_all = []
    test_tgt_all = []

    for year in test_years:
        test_src_file = os.path.join(data_dir, f"IWSLT17.TED.tst{year}.{language_pair}.{src_lang}.xml")
        test_tgt_file = os.path.join(data_dir, f"IWSLT17.TED.tst{year}.{language_pair}.{tgt_lang}.xml")

        if os.path.exists(test_src_file) and os.path.exists(test_tgt_file):
            src_sentences = parse_iwslt2017_xml_file(test_src_file)
            tgt_sentences = parse_iwslt2017_xml_file(test_tgt_file)

            # 确保对齐
            min_len = min(len(src_sentences), len(tgt_sentences))
            test_src_all.extend(src_sentences[:min_len])
            test_tgt_all.extend(tgt_sentences[:min_len])
            print(f"测试数据 {year}: {min_len} 个句子对")
        else:
            print(f"测试文件不存在: {test_src_file} 或 {test_tgt_file}")

    print(f"总测试数据: {len(test_src_all)} 个句子对")

    # 创建数据集字典
    dataset_dict = {
        "train": {
            src_lang: train_src,
            tgt_lang: train_tgt
        },
        "validation": {
            src_lang: dev_src,
            tgt_lang: dev_tgt
        },
        "test": {
            src_lang: test_src_all,
            tgt_lang: test_tgt_all
        }
    }

    # 转换为Hugging Face数据集格式
    train_dataset = Dataset.from_dict(dataset_dict["train"])
    val_dataset = Dataset.from_dict(dataset_dict["validation"])
    test_dataset = Dataset.from_dict(dataset_dict["test"])

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })


def load_huggingface_iwslt2017_data(language_pair: str = "en-de", cache_dir: str = "./data_cache") -> DatasetDict:
    """加载Hugging Face上的IWSLT2017数据集"""
    from datasets import load_dataset

    print(f"从Hugging Face加载IWSLT2017 {language_pair} 数据...")

    try:
        dataset = load_dataset("iwslt2017", f"iwslt2017-{language_pair}", cache_dir=cache_dir)
        print(f"成功加载Hugging Face数据: {dataset}")
        return dataset
    except Exception as e:
        print(f"从Hugging Face加载数据失败: {e}")
        print("请检查语言对是否可用或网络连接")
        return None


def build_vocabulary(sentences: List[str], max_vocab_size: int = 30000) -> Tuple[Dict[str, int], Dict[int, str]]:
    """构建词汇表"""
    word_counter = Counter()

    for sentence in sentences:
        # 简单的空格分词
        words = sentence.lower().split()
        word_counter.update(words)

    # 选择最常见的词
    most_common = word_counter.most_common(max_vocab_size - 4)  # 保留位置给特殊标记

    # 创建词汇表
    word_to_idx = {
        '<pad>': 0,  # 填充标记
        '<unk>': 1,  # 未知词
        '<bos>': 2,  # 开始标记
        '<eos>': 3  # 结束标记
    }

    idx_to_word = {v: k for k, v in word_to_idx.items()}

    # 添加词汇
    current_idx = 4
    for word, count in most_common:
        word_to_idx[word] = current_idx
        idx_to_word[current_idx] = word
        current_idx += 1

    print(f"词汇表大小: {len(word_to_idx)}")
    return word_to_idx, idx_to_word


def tokenize_and_encode(sentences: List[str], vocab: Dict[str, int], max_length: int = 100) -> List[List[int]]:
    """将句子转换为token IDs"""
    encoded_sentences = []

    for sentence in sentences:
        # 分词并转换为小写
        words = sentence.lower().split()

        # 添加开始和结束标记
        tokens = [vocab.get('<bos>', 2)]  # 开始标记

        for word in words:
            if len(tokens) >= max_length - 1:  # -1 为结束标记留位置
                break
            token = vocab.get(word, vocab.get('<unk>', 1))
            tokens.append(token)

        # 添加结束标记
        tokens.append(vocab.get('<eos>', 3))

        # 填充到固定长度
        while len(tokens) < max_length:
            tokens.append(vocab.get('<pad>', 0))

        encoded_sentences.append(tokens[:max_length])

    return encoded_sentences


class TranslationDataset(torch.utils.data.Dataset):
    """翻译数据集类"""

    def __init__(self, src_sentences: List[List[int]], tgt_sentences: List[List[int]]):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.src_sentences[idx], dtype=torch.long),
            torch.tensor(self.tgt_sentences[idx], dtype=torch.long)
        )


class TranslationDataLoaderFactory:
    """翻译数据加载器工厂"""

    @staticmethod
    def create_translation_data_loaders(
            dataset_name: str = 'iwslt2017',
            language_pair: str = 'en-de',
            max_length: int = 50,  # 减小默认长度
            batch_size: int = 128,  # 增大默认批次
            use_huggingface: bool = False,
            data_dir: str = './en-de',
            debug_mode: bool = True,  # 添加调试模式
            max_train_samples: int = 5000,  # 最大训练样本
            max_val_samples: int = 500,  # 最大验证样本
            max_vocab_size: int = 8000  # 最大词汇表大小
    ) -> Tuple[DataLoader, DataLoader, Dict, Dict, Dict]:
        """
        创建翻译数据加载器

        Returns:
            train_loader, val_loader, src_vocab, tgt_vocab, dataset_info
        """

        print(f"创建翻译数据加载器: {dataset_name}, {language_pair}")

        # 加载数据集
        if use_huggingface:
            dataset = load_huggingface_iwslt2017_data(language_pair)
            if dataset is None:
                raise ValueError("无法从Hugging Face加载数据")
        else:
            dataset = load_local_iwslt2017_data(data_dir, language_pair)

        src_lang, tgt_lang = language_pair.split('-')

        # 提取句子
        train_src_sentences = dataset['train'][src_lang]
        train_tgt_sentences = dataset['train'][tgt_lang]
        val_src_sentences = dataset['validation'][src_lang]
        val_tgt_sentences = dataset['validation'][tgt_lang]

        print(f"原始训练集: {len(train_src_sentences)} 句子对")
        print(f"原始验证集: {len(val_src_sentences)} 句子对")

        # 调试模式：限制数据量
        if debug_mode:
            print(f"调试模式：限制数据量加速训练")
            if len(train_src_sentences) > max_train_samples:
                train_src_sentences = train_src_sentences[:max_train_samples]
                train_tgt_sentences = train_tgt_sentences[:max_train_samples]
                print(f"限制训练数据到 {max_train_samples} 句")

            if len(val_src_sentences) > max_val_samples:
                val_src_sentences = val_src_sentences[:max_val_samples]
                val_tgt_sentences = val_tgt_sentences[:max_val_samples]
                print(f"限制验证数据到 {max_val_samples} 句")

        print(f"处理后训练集: {len(train_src_sentences)} 句子对")
        print(f"处理后验证集: {len(val_src_sentences)} 句子对")

        # 构建词汇表 - 使用限制的词汇表大小
        print("构建源语言词汇表...")
        src_vocab, src_idx_to_word = build_vocabulary(
            train_src_sentences + val_src_sentences,
            max_vocab_size=max_vocab_size
        )

        print("构建目标语言词汇表...")
        tgt_vocab, tgt_idx_to_word = build_vocabulary(
            train_tgt_sentences + val_tgt_sentences,
            max_vocab_size=max_vocab_size
        )

        # 编码句子
        print("编码训练数据...")
        train_src_encoded = tokenize_and_encode(train_src_sentences, src_vocab, max_length)
        train_tgt_encoded = tokenize_and_encode(train_tgt_sentences, tgt_vocab, max_length)

        print("编码验证数据...")
        val_src_encoded = tokenize_and_encode(val_src_sentences, src_vocab, max_length)
        val_tgt_encoded = tokenize_and_encode(val_tgt_sentences, tgt_vocab, max_length)

        # 创建数据集
        train_dataset = TranslationDataset(train_src_encoded, train_tgt_encoded)
        val_dataset = TranslationDataset(val_src_encoded, val_tgt_encoded)

        # 创建数据加载器 - 禁用多进程避免Windows问题
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Windows上设为0避免多进程问题
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Windows上设为0
            pin_memory=True if torch.cuda.is_available() else False
        )

        # 数据集信息
        dataset_info = {
            'name': dataset_name,
            'language_pair': language_pair,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'src_vocab_size': len(src_vocab),
            'tgt_vocab_size': len(tgt_vocab),
            'max_length': max_length
        }

        print(f"数据加载完成!")
        print(f"源语言词汇表大小: {len(src_vocab)}")
        print(f"目标语言词汇表大小: {len(tgt_vocab)}")
        print(f"训练批次数量: {len(train_loader)}")
        print(f"验证批次数量: {len(val_loader)}")

        return train_loader, val_loader, src_vocab, tgt_vocab, dataset_info


def print_translation_dataset_info():
    """打印翻译数据集信息"""
    print("=" * 60)
    print("IWSLT2017 翻译数据集信息")
    print("=" * 60)
    print("可用语言对: en-de, de-en, en-fr, fr-en, en-it, it-en")
    print("数据来源: TED演讲多语言翻译")
    print("数据规模: ~200k 平行句对 (英语-德语)")
    print("特点: 口语化文本，领域为技术、教育、文化等")
    print("本地数据格式:")
    print("  - 训练数据: train.tags.{lang-pair}.{lang}")
    print("  - 开发数据: IWSLT17.TED.dev2010.{lang-pair}.{lang}.xml")
    print("  - 测试数据: IWSLT17.TED.tst201[0-5].{lang-pair}.{lang}.xml")
    print("=" * 60)


def test_data_loading():
    """测试数据加载"""
    print("测试IWSLT2017数据加载...")

    try:
        # 测试本地数据加载
        train_loader, val_loader, src_vocab, tgt_vocab, dataset_info = \
            TranslationDataLoaderFactory.create_translation_data_loaders(
                dataset_name='iwslt2017',
                language_pair='en-de',
                max_length=50,
                batch_size=2,
                use_huggingface=False,
                data_dir='.\en-de'  # 修改为您的数据目录
            )

        print("数据加载测试成功!")
        print(f"数据集信息: {dataset_info}")

        # 测试一个批次
        for src, tgt in train_loader:
            print(f"源句子形状: {src.shape}")
            print(f"目标句子形状: {tgt.shape}")
            print(f"源句子示例: {src[0]}")
            print(f"目标句子示例: {tgt[0]}")
            break

    except Exception as e:
        print(f"数据加载测试失败: {e}")
        print("请检查数据目录和文件格式")


if __name__ == '__main__':
    print_translation_dataset_info()
    test_data_loading()