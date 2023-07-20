#!/bin/sh

CUDA_VISIBLE_DEVICES=5 python main.py \
        --nhid 32 \
        --patience 10 \
        --epoch 100 \
        --lr 1e-05 \
        --batch-size 4 \
        --nhead 4 \
        --loss pair \
        --v1 \
        --accelerator gpu \
        --check-val-freq 1 \
        --is-counter \
        --is-support \
        --run100 \
        # --test-ver 902 \
        # --fine-tune-we \