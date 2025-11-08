import torch.nn as nn
from torch.optim import AdamW
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math
import os
import json
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch
import torch.nn.functional as F
# 修复导入问题 - 确保能找到本地模块
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from translation_data_manager import TranslationDataLoaderFactory, TRANSLATION_DATASET_INFO
    from transformer_translation import TransformerForTranslation, create_translation_model
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 translation_data_manager.py 和 transformer_translation.py 在同一个目录")
    sys.exit(1)


class TranslationConfig:
    """翻译任务配置类"""

    def __init__(self):
        # 模型参数 - 大幅减小模型规模
        self.d_model = 64  # 进一步减小
        self.num_heads = 4
        self.num_encoder_layers = 2
        self.num_decoder_layers = 2
        self.d_ff = 256  # 进一步减小
        self.dropout = 0.1
        self.positional_encoding = 'sinusoidal'
        self.share_embeddings = False
        self.max_seq_length = 50  # 进一步减小序列长度

        # 训练参数 - 优化训练设置
        self.batch_size = 128  # 增大批次大小
        self.num_epochs = 5  # 减少epoch数
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.grad_clip = 1.0
        self.patience = 3  # 减少耐心值
        self.label_smoothing = 0.1

        # 数据参数
        self.dataset_name = 'iwslt2017'
        self.language_pair = 'en-de'  # 修复：正确的语言对格式
        self.use_huggingface = False
        self.data_dir = './en-de'  # 添加数据目录
        self.src_pad_idx = 0
        self.tgt_pad_idx = 0
        self.tgt_bos_idx = 2
        self.tgt_eos_idx = 3

        # 实验参数
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_dir = 'translation_results'
        self.cache_dir = './data_cache'
        self.enable_beam_search = False  # 禁用束搜索加速
        self.beam_size = 1

        # 调试参数
        self.debug_mode = True  # 启用调试模式限制数据
        self.max_train_samples = 5000  # 限制训练样本数
        self.max_val_samples = 500  # 限制验证样本数
        self.max_vocab_size = 8000  # 限制词汇表大小

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __str__(self) -> str:
        config_str = "翻译实验配置:\n"
        config_str += "=" * 50 + "\n"
        for key, value in self.to_dict().items():
            config_str += f"{key:25}: {value}\n"
        return config_str


class TranslationTrainer:
    """翻译训练器"""

    def __init__(self, model: nn.Module, train_loader, val_loader,
                 src_vocab: Dict, tgt_vocab: Dict, config: TranslationConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.config = config
        self.device = config.device

        # 反转词汇表用于解码
        self.src_idx_to_token = {v: k for k, v in src_vocab.items()}
        self.tgt_idx_to_token = {v: k for k, v in tgt_vocab.items()}

        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 损失函数（带标签平滑）
        if config.label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                config.tgt_vocab_size,
                smoothing=config.label_smoothing,
                ignore_index=config.tgt_pad_idx
            )
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=config.tgt_pad_idx)

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_ppls = []
        self.val_ppls = []
        self.learning_rates = []

        # 早停机制
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_tokens = 0

        progress_bar = tqdm(self.train_loader, desc=f'Training Epoch {epoch}')
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src, tgt = src.to(self.device), tgt.to(self.device)

            self.optimizer.zero_grad()

            # 前向传播 - 使用teacher forcing
            # 输入是tgt[:-1]，目标是tgt[1:]
            output = self.model(src, tgt[:, :-1],
                                self.config.src_pad_idx, self.config.tgt_pad_idx)

            loss = self.criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt[:, 1:].contiguous().view(-1)
            )

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()

            # 统计
            batch_tokens = (tgt[:, 1:] != self.config.tgt_pad_idx).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

            current_loss = loss.item()
            current_ppl = math.exp(current_loss)

            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'ppl': f'{current_ppl:.2f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

        return total_loss / total_tokens

    def validate(self) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for src, tgt in tqdm(self.val_loader, desc='Validation'):
                src, tgt = src.to(self.device), tgt.to(self.device)

                output = self.model(src, tgt[:, :-1],
                                    self.config.src_pad_idx, self.config.tgt_pad_idx)
                loss = self.criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt[:, 1:].contiguous().view(-1)
                )

                batch_tokens = (tgt[:, 1:] != self.config.tgt_pad_idx).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

        return total_loss / total_tokens

    def train(self) -> Dict[str, Any]:
        """完整训练过程"""
        print("开始翻译模型训练...")
        print(f"模型参数量: {self.model.count_parameters():,}")

        for epoch in range(1, self.config.num_epochs + 1):
            print(f'\nEpoch {epoch}/{self.config.num_epochs}')

            # 训练
            train_loss = self.train_epoch(epoch)
            train_ppl = math.exp(train_loss)

            # 验证
            val_loss = self.validate()
            val_ppl = math.exp(val_loss)

            # 记录结果
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_ppls.append(train_ppl)
            self.val_ppls.append(val_ppl)

            print(f'Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}')
            print(f'Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}')

            # 生成示例翻译
            if (epoch %1 == 0):

                self.generate_example_translations(epoch)

            # 检查早停
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
                print(f"保存最佳模型，验证损失: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, val_loss, is_best=False)

            if self.patience_counter >= self.config.patience:
                print(f"早停触发，在epoch {epoch}停止训练")
                break

            # 保存训练曲线
            if epoch % 10 == 0 or epoch == self.config.num_epochs:
                self.plot_training_curves()

        # 返回最终结果
        results = {
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_train_ppl': self.train_ppls[-1],
            'final_val_ppl': self.val_ppls[-1],
            'best_val_loss': self.best_val_loss,
            'parameters': self.model.count_parameters(),
            'total_epochs': len(self.train_losses)
        }

        print("训练完成!")
        return results



    def generate_example_translations(self, epoch: int, num_examples: int = 3):
        """生成示例翻译"""
        self.model.eval()

        print("\n示例翻译:")
        print("-" * 50)

        with torch.no_grad():
            # 获取一个批次的数据
            src_batch, tgt_batch = next(iter(self.val_loader))
            src_batch = src_batch[:num_examples].to(self.device)
            tgt_batch = tgt_batch[:num_examples].to(self.device)

            # 生成翻译
            generated = self.model.generate(
                src_batch,
                max_len=self.config.max_seq_length,
                src_pad_idx=self.config.src_pad_idx,
                tgt_bos_idx=self.config.tgt_bos_idx,
                tgt_eos_idx=self.config.tgt_eos_idx
            )

            for i in range(num_examples):
                src_tokens = src_batch[i].cpu().tolist()
                tgt_tokens = tgt_batch[i].cpu().tolist()
                gen_tokens = generated[i].cpu().tolist()

                # 解码文本
                src_text = self.decode_tokens(src_tokens, self.src_idx_to_token)
                tgt_text = self.decode_tokens(tgt_tokens, self.tgt_idx_to_token)
                gen_text = self.decode_tokens(gen_tokens, self.tgt_idx_to_token)

                print(f"源文: {src_text}")
                print(f"目标: {tgt_text}")
                print(f"生成: {gen_text}")
                print("-" * 30)

    def decode_tokens(self, tokens: List[int], idx_to_token: Dict[int, str]) -> str:
        """解码tokens为文本"""
        # 移除特殊标记和填充
        filtered_tokens = []
        for token in tokens:
            if token in [self.config.tgt_pad_idx, self.config.tgt_bos_idx]:
                continue
            if token == self.config.tgt_eos_idx:
                break
            filtered_tokens.append(token)

        # 转换为文本
        words = []
        for token in filtered_tokens:
            if token in idx_to_token:
                words.append(idx_to_token[token])
            else:
                words.append('<unk>')

        return " ".join(words)

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_ppls': self.train_ppls,
            'val_ppls': self.val_ppls,
            'config': self.config.to_dict(),
            'src_vocab': self.src_vocab,
            'tgt_vocab': self.tgt_vocab
        }

        os.makedirs(self.config.save_dir, exist_ok=True)

        if is_best:
            torch.save(checkpoint, f'{self.config.save_dir}/best_model.pth')
        torch.save(checkpoint, f'{self.config.save_dir}/checkpoint_epoch_{epoch}.pth')

    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(15, 5))

        # 损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.grid(True, alpha=0.3)

        # 困惑度曲线
        plt.subplot(1, 3, 2)
        plt.plot(self.train_ppls, label='Train PPL', linewidth=2)
        plt.plot(self.val_ppls, label='Val PPL', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.title('Training and Validation Perplexity')
        plt.grid(True, alpha=0.3)

        # 学习率曲线
        plt.subplot(1, 3, 3)
        plt.plot(self.learning_rates, color='purple', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.config.save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""

    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(output, dim=-1)

        with torch.no_grad():
            smooth_label = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 1))
            smooth_label.scatter_(-1, target.unsqueeze(-1), self.confidence)
            mask = (target == self.ignore_index).unsqueeze(-1)
            smooth_label.masked_fill_(mask, 0)

        return (-smooth_label * log_probs).sum(dim=-1).mean()


class TranslationExperiment:
    """翻译实验运行器"""

    def __init__(self):
        self.config = TranslationConfig()

    def run_baseline_experiment(self) -> Dict[str, Any]:
        """运行基线翻译实验"""

        print("运行基线翻译实验...")
        print(self.config)

        # 获取数据
        train_loader, val_loader, src_vocab, tgt_vocab, dataset_info = \
            TranslationDataLoaderFactory.create_translation_data_loaders(
                dataset_name=self.config.dataset_name,
                language_pair=self.config.language_pair,
                max_length=self.config.max_seq_length,
                batch_size=self.config.batch_size,
                use_huggingface=self.config.use_huggingface
            )

        print(f"数据集信息: {dataset_info}")
        print(f"源语言词汇表大小: {len(src_vocab)}")
        print(f"目标语言词汇表大小: {len(tgt_vocab)}")

        # 更新配置
        self.config.src_vocab_size = len(src_vocab)
        self.config.tgt_vocab_size = len(tgt_vocab)

        # 创建模型
        model = create_translation_model(len(src_vocab), len(tgt_vocab), self.config)
        model = model.to(self.config.device)

        # 训练
        trainer = TranslationTrainer(model, train_loader, val_loader, src_vocab, tgt_vocab, self.config)
        results = trainer.train()

        return results, src_vocab, tgt_vocab

    def run_ablation_study(self):
        """运行翻译消融实验"""

        base_config = self.config
        results = {}

        # 实验配置
        experiments = {
            'baseline': {
                'description': '完整Transformer翻译模型',
                'config_updates': {}
            },
            'no_positional_encoding': {
                'description': '移除位置编码',
                'config_updates': {'positional_encoding': 'none'}
            },
            'single_head_attention': {
                'description': '单头注意力',
                'config_updates': {'num_heads': 1}
            },
            'shallow_encoder': {
                'description': '浅编码器',
                'config_updates': {'num_encoder_layers': 1}
            },
            'shallow_decoder': {
                'description': '浅解码器',
                'config_updates': {'num_decoder_layers': 1}
            },
            'small_model': {
                'description': '小模型',
                'config_updates': {'d_model': 128, 'd_ff': 512}
            },
            'shared_embeddings': {
                'description': '共享词嵌入',
                'config_updates': {'share_embeddings': True}
            }
        }

        for exp_name, exp_config in experiments.items():
            print(f"\n{'=' * 60}")
            print(f"运行翻译消融实验: {exp_name}")
            print(f"描述: {exp_config['description']}")
            print(f"{'=' * 60}")

            # 更新配置
            current_config = TranslationConfig()
            for key, value in base_config.to_dict().items():
                setattr(current_config, key, value)

            for key, value in exp_config['config_updates'].items():
                setattr(current_config, key, value)

            # 设置实验特定的保存目录
            current_config.save_dir = f"{base_config.save_dir}/ablation_{exp_name}"
            current_config.num_epochs = 1  # 消融实验用较少的epoch

            # 获取数据
            train_loader, val_loader, src_vocab, tgt_vocab, dataset_info = \
                TranslationDataLoaderFactory.create_translation_data_loaders(
                    dataset_name=current_config.dataset_name,
                    language_pair=current_config.language_pair,
                    max_length=current_config.max_seq_length,
                    batch_size=current_config.batch_size,
                    use_huggingface=current_config.use_huggingface
                )

            # 更新词汇表大小
            current_config.src_vocab_size = len(src_vocab)
            current_config.tgt_vocab_size = len(tgt_vocab)

            # 创建模型
            model = create_translation_model(len(src_vocab), len(tgt_vocab), current_config)
            model = model.to(current_config.device)

            # 训练
            trainer = TranslationTrainer(model, train_loader, val_loader, src_vocab, tgt_vocab, current_config)
            result = trainer.train()

            # 记录结果
            results[exp_name] = {
                **result,
                'description': exp_config['description'],
                'config': current_config.to_dict(),
                'dataset_info': dataset_info
            }

            print(f"完成: {exp_name} - 最终验证损失: {result['final_val_loss']:.4f}")

        return results

    def analyze_translation_results(self, results: Dict[str, Any]):
        """分析翻译实验结果"""

        import pandas as pd

        # 创建结果表格
        results_data = []
        baseline_loss = results['baseline']['final_val_loss']

        for exp_name, result in results.items():
            performance_drop = result['final_val_loss'] - baseline_loss
            relative_drop = (performance_drop / baseline_loss) * 100

            results_data.append({
                'Experiment': exp_name,
                'Description': result['description'],
                'Val Loss': result['final_val_loss'],
                'Val PPL': result['final_val_ppl'],
                'Parameters': result['parameters'],
                'Performance Drop': performance_drop,
                'Relative Drop (%)': relative_drop
            })

        df = pd.DataFrame(results_data)
        df = df.sort_values('Val Loss')

        print("\n翻译消融实验结果:")
        print("=" * 80)
        print(df.to_string(index=False))

        # 可视化结果
        self.visualize_translation_results(results, df)

        return df

    def visualize_translation_results(self, results: Dict[str, Any], df: pd.DataFrame):
        """可视化翻译实验结果"""

        os.makedirs(f"{self.config.save_dir}/translation_analysis", exist_ok=True)

        # 创建可视化
        plt.figure(figsize=(16, 10))

        # 1. 性能对比
        plt.subplot(2, 3, 1)
        experiments = list(results.keys())
        val_losses = [results[exp]['final_val_loss'] for exp in experiments]

        colors = ['orange' if exp == 'baseline' else 'skyblue' for exp in experiments]
        bars = plt.bar(experiments, val_losses, color=colors)
        plt.title('Validation Loss Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Loss')
        plt.xticks(rotation=45, ha='right')

        for bar, value in zip(bars, val_losses):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{value:.3f}', ha='center', va='bottom')

        # 2. 性能下降分析
        plt.subplot(2, 3, 2)
        performance_drops = {}
        baseline_loss = results['baseline']['final_val_loss']

        for exp_name, result in results.items():
            if exp_name != 'baseline':
                performance_drops[exp_name] = result['final_val_loss'] - baseline_loss

        sorted_drops = dict(sorted(performance_drops.items(), key=lambda x: x[1], reverse=True))

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_drops)))
        bars = plt.barh(list(sorted_drops.keys()), list(sorted_drops.values()), color=colors)
        plt.title('Performance Drop vs Baseline', fontsize=14, fontweight='bold')
        plt.xlabel('Increase in Validation Loss')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)

        for bar, value in zip(bars, sorted_drops.values()):
            plt.text(value + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'+{value:.3f}', ha='left', va='center')

        # 3. 组件重要性分析
        plt.subplot(2, 3, 3)
        component_impact = {
            'Positional Encoding': results['no_positional_encoding']['final_val_loss'] - baseline_loss,
            'Multi-Head Attention': results['single_head_attention']['final_val_loss'] - baseline_loss,
            'Deep Encoder': results['shallow_encoder']['final_val_loss'] - baseline_loss,
            'Deep Decoder': results['shallow_decoder']['final_val_loss'] - baseline_loss,
            'Model Size': results['small_model']['final_val_loss'] - baseline_loss
        }

        components = list(component_impact.keys())
        impacts = list(component_impact.values())

        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(components)))
        bars = plt.barh(components, impacts, color=colors)
        plt.title('Component Importance in Translation', fontsize=14, fontweight='bold')
        plt.xlabel('Performance Impact (Loss Increase)')

        for bar, impact in zip(bars, impacts):
            plt.text(impact + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'+{impact:.3f}', ha='left', va='center')

        plt.tight_layout()
        plt.savefig(f'{self.config.save_dir}/translation_analysis/translation_results.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 保存结果
        df.to_csv(f'{self.config.save_dir}/translation_analysis/translation_results.csv', index=False)

        # 生成报告
        self.generate_translation_report(results, df)

    def generate_translation_report(self, results: Dict[str, Any], df: pd.DataFrame):
        """生成翻译实验报告"""

        report = {
            'summary': {
                'total_experiments': len(results),
                'best_experiment': df.iloc[0]['Experiment'],
                'worst_experiment': df.iloc[-1]['Experiment'],
                'baseline_performance': {
                    'val_loss': results['baseline']['final_val_loss'],
                    'val_ppl': results['baseline']['final_val_ppl']
                },
                'dataset': self.config.dataset_name,
                'language_pair': self.config.language_pair
            },
            'key_findings': self._extract_translation_findings(results, df),
            'recommendations': self._generate_translation_recommendations(results, df),
            'detailed_results': df.to_dict('records')
        }

        # 保存JSON报告
        with open(f'{self.config.save_dir}/translation_analysis/translation_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        # 生成Markdown报告
        self._generate_translation_markdown_report(report)

    def _extract_translation_findings(self, results: Dict[str, Any], df: pd.DataFrame) -> List[str]:
        """提取翻译实验关键发现"""

        findings = []
        baseline_loss = results['baseline']['final_val_loss']

        for _, row in df.iterrows():
            if row['Experiment'] != 'baseline':
                relative_drop = row['Relative Drop (%)']

                if relative_drop > 50:
                    findings.append(f"移除{row['Experiment']}导致翻译性能严重下降({relative_drop:.1f}%)")
                elif relative_drop > 20:
                    findings.append(f"移除{row['Experiment']}对翻译性能有显著影响({relative_drop:.1f}%)")
                elif relative_drop < 5:
                    findings.append(f"移除{row['Experiment']}对翻译性能影响较小({relative_drop:.1f}%)")

        # 分析最佳配置
        best_exp = df.iloc[0]
        if best_exp['Experiment'] != 'baseline':
            findings.append(f"最佳翻译配置是{best_exp['Experiment']}，比baseline性能更好")

        # 特定发现
        if results['shared_embeddings']['final_val_loss'] < baseline_loss:
            findings.append("共享词嵌入在英德翻译任务上表现更好")

        return findings

    def _generate_translation_recommendations(self, results: Dict[str, Any], df: pd.DataFrame) -> List[str]:
        """生成翻译优化建议"""

        recommendations = []
        baseline_params = results['baseline']['parameters']

        # 分析参数效率
        for _, row in df.iterrows():
            if row['Parameters'] < baseline_params * 0.7 and row['Val Loss'] < results['baseline'][
                'final_val_loss'] * 1.1:
                recommendations.append(
                    f"{row['Experiment']}在保持翻译性能的同时减少了{(1 - row['Parameters'] / baseline_params) * 100:.1f}%参数量"
                )

        # 组件特定建议
        if results['no_positional_encoding']['final_val_loss'] > results['baseline']['final_val_loss'] * 1.3:
            recommendations.append("位置编码对机器翻译任务至关重要")

        if results['shared_embeddings']['final_val_loss'] < results['baseline']['final_val_loss']:
            recommendations.append("在相关语言对（如英德）上共享词嵌入可以提高性能")

        if results['single_head_attention']['final_val_loss'] > results['baseline']['final_val_loss'] * 1.2:
            recommendations.append("多头注意力机制对翻译质量有重要影响")

        return recommendations

    def _generate_translation_markdown_report(self, report: Dict):
        """生成翻译Markdown报告"""

        md_content = f"""# Transformer机器翻译实验报告

## 实验概述

- **任务**: 机器翻译 ({report['summary']['language_pair']})
- **数据集**: {report['summary']['dataset']}
- **总实验数**: {report['summary']['total_experiments']}
- **最佳配置**: `{report['summary']['best_experiment']}`
- **基准性能**: 损失={report['summary']['baseline_performance']['val_loss']:.4f}, 困惑度={report['summary']['baseline_performance']['val_ppl']:.2f}

## 详细结果

| 实验 | 描述 | 验证损失 | 验证困惑度 | 参数量 | 性能下降 |
|------|------|----------|------------|--------|----------|
"""

        for result in report['detailed_results']:
            md_content += f"| {result['Experiment']} | {result['Description']} | {result['Val Loss']:.4f} | {result['Val PPL']:.2f} | {result['Parameters']:,} | {result['Performance Drop']:+.4f} |\n"

        md_content += """

## 关键发现

"""

        for finding in report['key_findings']:
            md_content += f"- {finding}\n"

        md_content += """

## 优化建议

"""

        for recommendation in report['recommendations']:
            md_content += f"- {recommendation}\n"

        md_content += """

## 结论

基于IWSLT2017数据集的机器翻译实验表明，完整的Encoder-Decoder Transformer架构在英德翻译任务上表现良好。
关键组件按重要性排序为：位置编码 > 多头注意力 > 编码器深度 > 解码器深度 > 模型尺寸。
"""

        with open(f'{self.config.save_dir}/translation_analysis/translation_report.md', 'w') as f:
            f.write(md_content)

    def run_full_translation_experiment(self):
        """运行完整翻译实验"""

        print("=" * 60)
        print("Transformer机器翻译实验系统")
        print("=" * 60)

        # 创建保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config.save_dir = f"translation_results_{timestamp}"
        os.makedirs(self.config.save_dir, exist_ok=True)

        # 打印数据集信息
        from translation_data_manager import print_translation_dataset_info
        print_translation_dataset_info()

        # 运行基线实验
        # print("\n1. 运行基线翻译实验...")
        # baseline_results, src_vocab, tgt_vocab = self.run_baseline_experiment()

        # 运行消融实验
        print("\n2. 运行翻译消融实验...")
        ablation_results = self.run_ablation_study()

        # 分析结果
        print("\n3. 分析翻译实验结果...")
        results_df = self.analyze_translation_results(ablation_results)

        print("\n" + "=" * 60)
        print("翻译实验完成!")
        print("=" * 60)
        print(f"结果保存在: {self.config.save_dir}/")
        print(f"- 训练曲线: {self.config.save_dir}/training_curves.png")
        print(f"- 最佳模型: {self.config.save_dir}/best_model.pth")
        print(f"- 翻译分析: {self.config.save_dir}/translation_analysis/")
        print(f"- 实验报告: {self.config.save_dir}/translation_analysis/translation_report.md")


def main():
    """主函数"""

    # 创建翻译实验运行器
    experiment = TranslationExperiment()

    # 配置实验参数
    experiment.config.dataset_name = 'iwslt2017'
    experiment.config.language_pair = 'en-de'
    experiment.config.use_huggingface = False  # 使用本地数据集避免网络问题
    experiment.config.num_epochs = 50
    experiment.config.d_model = 256
    experiment.config.num_encoder_layers = 3
    experiment.config.num_decoder_layers = 3

    # 运行完整实验
    experiment.run_full_translation_experiment()


if __name__ == '__main__':

    main()