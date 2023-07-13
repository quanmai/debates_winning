#!/bin/sh

CUDA_VISIBLE_DEVICES=5 python main.py \
        --nhid 32 \
        --patience 10 \
        --epoch 100 \
        --lr 1e-05 \
        --batch-size 8 \
        --nhead 4 \
        --loss pair \
        --v1 \
        --accelerator gpu \
        # --fine-tune-we \
