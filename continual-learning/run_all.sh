#! bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py --ewc --lambda 1000 --iters 50 --batch 64 --singlehead --experiment permMNIST --seed 0 --date 190428 --scenario task &
CUDA_VISIBLE_DEVICES=1 python main.py --ewc --lambda 3000 --iters 50 --batch 64 --singlehead --experiment permMNIST --seed 0 --date 190428 --scenario task&
CUDA_VISIBLE_DEVICES=2 python main.py --ewc --lambda 5000 --iters 50 --batch 64 --singlehead --experiment permMNIST --seed 0 --date 190428 --scenario task&
CUDA_VISIBLE_DEVICES=3 python main.py --ewc --lambda 10000 --iters 50 --batch 64 --singlehead --experiment permMNIST --seed 0 --date 190428 --scenario task&

CUDA_VISIBLE_DEVICES=0 python main.py --ewc --online --lambda 3000 --iters 50 --batch 64 --singlehead --experiment permMNIST --seed 0 --date 190428 --scenario task&
CUDA_VISIBLE_DEVICES=1 python main.py --ewc --online --lambda 1000 --iters 50 --batch 64 --singlehead --experiment permMNIST --seed 0 --date 190428 --scenario task&
CUDA_VISIBLE_DEVICES=2 python main.py --ewc --online --lambda 500 --iters 50 --batch 64 --singlehead --experiment permMNIST --seed 0 --date 190428 --scenario task&
CUDA_VISIBLE_DEVICES=3 python main.py --ewc --online --lambdd 200 --iters 50 --batch 64 --singlehead --experiment permMNIST --seed 0 --date 190428 --scenario task&

CUDA_VISIBLE_DEVICES=0 python main.py --si --c 0.1 --iters 50 --batch 64 --singlehead --experiment permMNIST --seed 0 --date 190428 --scenario task&
CUDA_VISIBLE_DEVICES=1 python main.py --si --c 0.3 --iters 50 --batch 64 --singlehead --experiment permMNIST --seed 0 --date 190428 --scenario task&
CUDA_VISIBLE_DEVICES=2 python main.py --si --c 0.6 --iters 50 --batch 64 --singlehead --experiment permMNIST --seed 0 --date 190428 --scenario task&
CUDA_VISIBLE_DEVICES=3 python main.py --si --c 1.0 --iters 50 --batch 64 --singlehead --experiment permMNIST --seed 0 --date 190428 --scenario task&