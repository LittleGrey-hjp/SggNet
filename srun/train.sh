#!/bin/bash
#SBATCH -J COD #作业名字（⾮必需）
#SBATCH -o %j.out.txt #标准输出⽂件（⾮必需）
#SBATCH -e %j.err.txt #标准错误输出⽂件（⾮必需）
#SBATCH -p gpu #运⾏分区（⾮必需，如不指定分配到默认分区）
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

source /share/home/jphe/anaconda3/etc/profile.d/conda.sh
conda activate torch_mmcd

echo "start $(date)"
### the command to run
srun --nodelist=c02 --pty python /share/home/jphe/project/SggNet/train_SggNet.py --path "/share/home/byliu/data/EORSSD/train/" --pretrain "/share/home/jphe/project/SggNet/mobilenet_v2-b0353104.pth"
echo "end $(date)"
