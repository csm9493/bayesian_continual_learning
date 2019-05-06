#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python run_permuted.py --date 190503 --trial 4 --batch 64 --tasknum 10 --epochs 50 &
CUDA_VISIBLE_DEVICES=1 python run_permuted.py --date 190503 --trial 4 --batch 64 --tasknum 10 --epochs 100 &

CUDA_VISIBLE_DEVICES=2 python run_permuted.py --date 190503 --trial 4 --batch 128 --tasknum 10 --epochs 100 &
CUDA_VISIBLE_DEVICES=3 python run_permuted.py --date 190503 --trial 4 --batch 128 --tasknum 10 --epochs 200 &

CUDA_VISIBLE_DEVICES=1 python run_permuted.py --date 190503 --trial 4 --batch 256 --tasknum 10 --epochs 100 &
CUDA_VISIBLE_DEVICES=2 python run_permuted.py --date 190503 --trial 4 --batch 256 --tasknum 10 --epochs 200 &

# CUDA_VISIBLE_DEVICES=1 python run_permuted.py --date 190423 --trial 2 &
# CUDA_VISIBLE_DEVICES=2 python run_permuted.py --date 190423 --trial 3 &
# CUDA_VISIBLE_DEVICES=3 python run_permuted.py --date 190423 --trial 4 &
# CUDA_VISIBLE_DEVICES=1 python run_permuted.py --date 190423 --trial 5 &