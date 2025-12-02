#!/bin/bash
# 简单的SLURM提交脚本

# 激活conda环境（如果需要）
# source /data/apps/miniconda3/24.9.2/etc/profile.d/conda.sh
conda activate qwen7b-ft

# 或者使用uv环境（如果使用uv管理Python环境）
# source .venv/bin/activate

# 检查Python环境
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
nvidia-smi

# 进入工作目录
cd /data/home/scyb494/meowmeow/final7008

export https_proxy=http://172.16.21.32:3128
export http_proxy=http://172.16.21.32:3128

export HF_ENDPOINT=https://hf-mirror.com

# 运行训练脚本
echo "开始执行..."
python main.py --model qwen --task "Use https://www.sogou.com to search for the latest news about AI advancements and summarize the key points." 


# echo "开始执行Qwen-7B-VL微调..."
# python train_fullft_qwen2vl7b.py
