#!/bin/bash
#
#SBATCH --job-name=train # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00 # set this time according to your need
#SBATCH --mem=64GB # how much RAM will your notebook consume? 
#SBATCH --gres=gpu:1 # if you need to use a GPU
#SBATCH -p simurgh # specify partition
#SBATCH --account=simurgh
#SBATCH --exclude=simurgh2
#SBATCH -o ./job_out/%j-pretrain.out
#SBATCH -e ./job_err/%j-pretrain.err \ 

source activate keymorph

python /sailhome/alanqw/rssl/scripts/run.py \
    -c /afs/cs.stanford.edu/u/alanqw/rssl/configs/config_randominit_group.json \
    -e /afs/cs.stanford.edu/u/alanqw/rssl/configs/environment.json \
    --run_mode train \
    --dataset_type adni \
    --visualize \
    --affine_slope -1 \
    --max_random_affine_augment_params 0.1 0.1 0.25 0.1 \
    --max_random_affine_augment_params_pair 0.0 0.0 0.0 0.0 \
    --log_interval 25 \
    # --use_checkpoint \
    # --anneal_alpha 500
    # --train_dataset gigamed_skullstripped_samemod \
    # --max_random_affine_augment_params 0.2 0.15 3.1416 0.1
    # --use_ema_encoder \
    # --resume_latest
    # --as_regularization \
    # --load_path /midtier/sablab/scratch/alw4013/keymorph/experiments/conv/__pretraining__gigamed-synthbrain-randomanisotropy_datasetgigamed+synthbrain+randomanisotropy_keypoints${NUM_KEY}_batch1_normTypeinstance_lr0.0001/checkpoints/pretrained_epoch8500_model.pth.tar