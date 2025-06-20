
# SMALLEST POSSIBLE ALLOCATION

# a100
srun --partition=gpu_a100 --ntasks=1 --gpus=1 --cpus-per-task=18 --time=01:00:00 --pty bash -i

# h100
srun --partition=gpu_h100 --ntasks=1 --gpus=1 --cpus-per-task=16 --time=01:00:00 --pty bash -i

# FULL ALLOCATION

# genoa
srun --partition=genoa --ntasks=1 --cpus-per-task=192 --time=01:00:00 --pty bash -i

# a100
srun --partition=gpu_a100 --ntasks=1 --gpus=4 --cpus-per-task=72 --time=01:00:00 --pty bash -i

# h100
srun --partition=gpu_h100 --ntasks=1 --gpus=4 --cpus-per-task=64 --time=01:00:00 --pty bash -i

module purge
module load 2024
module load Anaconda3/2024.06-1

source activate hycoclip

ssh -L 5901:tcn679:5901 scur1160@snellius.surf.nl


# generate trees
python hierchical_datasets/preprocess.py \
    --dataset imagenet \
    --imagenet_path ./imagenet \
    --output_dir hierchical_datasets/ \
    --checkpoint_path ./checkpoints/hycoclip_vit_b.pth \
    --train_config ./hycoclip/configs/train_hycoclip_vit_b.py \
    --limit 200

python hierchical_datasets/preprocess.py \
    --dataset grit \
    --grit_path /scratch-shared/grit/processed \
    --output_dir hierchical_datasets/ \
    --checkpoint_path ./checkpoints/hycoclip_vit_b.pth \
    --train_config ./hycoclip/configs/train_hycoclip_vit_b.py \
    --limit 200


# generate projections afterwards
python projection_methods/create_projections.py \
    --dataset-path hierchical_datasets/GRIT \
    --methods horopca \
    --n-project 200 \
    --plot 

python projection_methods/create_projections.py \
    --dataset-path hierchical_datasets/GRIT \
    --methods cosne \
    --n-project 200 \
    --plot 

python projection_methods/create_projections.py \
    --dataset-path hierchical_datasets/ImageNet \
    --methods horopca \
    --n-project 200 \
    --plot 

python projection_methods/create_projections.py \
    --dataset-path hierchical_datasets/ImageNet \
    --methods cosne \
    --n-project 200 \
    --plot 