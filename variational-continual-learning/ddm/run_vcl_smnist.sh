#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python run_split.py --date 190507 --trial 1 &
CUDA_VISIBLE_DEVICES=1 python run_split.py --date 190507 --trial 1 --singlehead &
