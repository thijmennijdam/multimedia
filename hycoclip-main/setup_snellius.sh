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
huggingface-cli download avik-pal/hycoclip hycoclip_vit_b.pth --local-dir ./checkpoints



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


# LOAD DATA
./install_git_lfs.sh
git clone https://huggingface.co/datasets/zzliang/GRIT

pip install img2dataset

export NO_ALBUMENTATIONS_UPDATE=1

# download dataset
img2dataset --url_list GRIT/grit-20m --input_format "parquet"\
    --url_col "url" --caption_col "caption" --output_format webdataset \
    --output_folder /scratch-shared/grit --processes_count 4 --thread_count 64 --image_size 256 \
    --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True \
    --save_additional_columns '["id","noun_chunks","ref_exps","clip_similarity_vitb32","clip_similarity_vitl14"]' \
    --enable_wandb False


# preprate dataset
python utils/prepare_GRIT_webdataset.py --raw_webdataset_path  /scratch-shared/grit \
    --processed_webdataset_path  /scratch-shared/grit/processed \
    --max_num_processes 12



# visualize embeddings

python scripts/visualize_embeddings.py --checkpoint-path checkpoints/hycoclip_vit_s.pth \
    --train-config configs/train_hycoclip_vit_s.py


python scripts/visualize_embeddings.py --checkpoint-path checkpoints/hycoclip_vit_b.pth \
    --train-config configs/train_hycoclip_vit_b.py



python scripts/projection_methods_analysis.py \
    --checkpoint-path checkpoints/hycoclip_vit_b.pth --train-config configs/train_hycoclip_vit_b.py \
    --methods horopca \
    --n-embed 200 \
    --n-project 200

python scripts/projection_methods_analysis.py \
    --checkpoint-path checkpoints/hycoclip_vit_b.pth --train-config configs/train_hycoclip_vit_b.py \
    --methods horopca \
    --n-embed 1000 \
    --n-project 1000

python scripts/projection_methods_analysis.py \
    --checkpoint-path checkpoints/hycoclip_vit_b.pth --train-config configs/train_hycoclip_vit_b.py \
    --methods cosne \
    --n-embed 1000 \
    --n-project 1000 \
    --cosne-perplexity 999 


python scripts/projection_methods_analysis.py \
    --checkpoint-path checkpoints/hycoclip_vit_b.pth --train-config configs/train_hycoclip_vit_b.py \
    --methods umap \
    --umap-neighbors 100 \
    --n-embed 1000 \
    --n-project 1000 \
    --umap-components