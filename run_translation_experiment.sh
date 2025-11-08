#!/bin/bash
# run_translation_experiment.sh

echo "=== Transformer机器翻译实验 ==="

# 创建环境
echo "创建Python环境..."
python -m venv translation_env
source translation_env/bin/activate  # Linux/Mac
# translation_env\Scripts\activate  # Windows

# 安装依赖
echo "安装依赖..."
pip install torch torchvision torchaudio
pip install matplotlib numpy pandas seaborn tqdm sentencepiece

# 创建目录
mkdir -p translation_results

# 运行基线翻译实验
echo "运行基线翻译实验..."
python run_translation.py --dataset iwslt2017 --language-pair en-de --epochs 30

# 运行完整翻译实验（包含消融）
echo "运行完整翻译实验..."
python run_translation.py --dataset iwslt2017 --language-pair en-de --epochs 50 --ablation

# 运行Hugging Face数据集实验（需要网络）
# echo "运行Hugging Face翻译实验..."
# python run_translation.py --dataset iwslt2017 --use-huggingface --epochs 30

echo "翻译实验完成！结果保存在 translation_results/ 目录"