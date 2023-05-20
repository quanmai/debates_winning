#!/bin/sh

# CUDA_VISIBLE_DEVICES=6 python main.py \
#                     --batch-size 8 \
#                     --nhid 16 \
#                     --patience 10 \
#                     --epoch 1000 \
#                     --is-counter \
#                     --is-support \
#                     --accelerator gpu \
#                     --run100 \
#                     --lr 5e-04 \
#                     --loss binary \
#                     --optimizer sgd \
#                     # --mode bidirection \
#                     # --scheduler exp \
#                     # --sparsify threshold \
CUDA_VISIBLE_DEVICES=6 python main.py \
                    --batch-size 128 \
                    --nhid 16 \
                    --patience 200 \
                    --accelerator gpu \
                    --epoch 1000 \
                    --is-counter \
                    --is-support \
                    --lr 5e-04\
                    --loss binary \
                    --scheduler exp \
                    # --mode bidirection \
                    # --nhead 1 \
                    # --optimizer sgd \
                    # --sparsify threshold \
