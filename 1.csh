#!/bin/sh

CUDA_VISIBLE_DEVICES=6 python main.py \
        --nhid 32 \
        --patience 10 \
        --epoch 100 \
        --lr 1e-05 \
        --batch-size 4 \
        --nhead 4 \
        --loss pair \
        --v1 \
        --accelerator gpu \
        --run100 \
        --loss ranking \
        --check-val-freq 1 \
        # --test-ver 902 \
        # --fine-tune-we \
        # CUDA_VISIBLE_DEVICES=5 
