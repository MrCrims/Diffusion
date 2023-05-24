#!/bin/bash

#SBATCH -p gpu4
#SBATCH -N 1-1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres gpu:1
#SBATCH -o diffusion-%j.out # 注意可以修改"slurm"为与任务相关的内容方便以后查询实验结果
#SBATCH --mem 20G


date
# 下面的 test_gpu.py 换成你的 Python 脚本路径
srun python train.py
srun python mail.py -t "DIFFUSION"
date