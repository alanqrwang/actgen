#!/bin/bash
#
#SBATCH --job-name=train # give your job a name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00 # set this time according to your need
#SBATCH --mem=128GB # how much RAM will your notebook consume? 
#SBATCH --gres=gpu:1 # if you need to use a GPU
#SBATCH -p simurgh # specify partition
#SBATCH --account=simurgh
#SBATCH --exclude=simurgh2
#SBATCH -o ./job_out/%j-train.out
#SBATCH -e ./job_err/%j-train.err \ 

source activate keymorph

<<<<<<< HEAD
python /sailhome/alanqw/actgen/scripts/run.py \
    -c $1 \
    -e /afs/cs.stanford.edu/u/alanqw/actgen/configs/environment.json
=======
python /sailhome/alanqw/rssl/scripts/run.py \
    -c $1 \
    -e /afs/cs.stanford.edu/u/alanqw/rssl/configs/environment.json
>>>>>>> 9abb11707ca6ada73f8c722c6f7f99edbf347d65
