#!/bin/sh

# CUDA_VISIBLE_DEVICES=6 python main.py --batch-size 10 --nhid 64 --patience 5 --epoch 30 --is-counter --is-support --scheduler exp --run100 --accelerator gpu  #--loss binary
CUDA_VISIBLE_DEVICES=6 python main.py --batch-size 100 --nhid 64 --patience 5 --accelerator gpu --epoch 50 --is-counter --is-support --scheduler exp --lr 0.0005 #--loss binary
