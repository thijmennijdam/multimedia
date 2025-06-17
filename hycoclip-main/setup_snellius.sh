srun --partition=gpu_a100 --ntasks=1 --gpus=1 --cpus-per-task=16 --time=01:00:00 --pty bash -i


# SETUP
cd hycoclip-main

module purge
module load 2024
module load Anaconda3/2024.06-1

conda create -n hycoclip python=3.9 --yes
source activate hycoclip

hugginface-cli login

# ENTER TOKEN


huggingface-cli download avik-pal/hycoclip hycoclip_vit_s.pth --local-dir ./checkpoints



# RUN AFTER SETUP
srun --partition=gpu_a100 --ntasks=1 --gpus=1 --cpus-per-task=16 --time=01:00:00 --pty bash -i

cd hycoclip-main

module purge
module load 2024
module load Anaconda3/2024.06-1

source activate hycoclip

python scripts/evaluate.py --config configs/eval_zero_shot_classification.py \
    --checkpoint-path checkpoints/hycoclip_vit_s.pth \
    --train-config configs/train_hycoclip_vit_s.py