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
CUDA_VISIBLE_DEVICES=5 python main.py \
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
                    --sparsify threshold \
                    --node-encoder-direction out \
                    # --gat-layers 2 \
                    # --counter-coeff 0.8 \
                    # --mode bidirection \
                    # --test-ver 462 \
                    # --nhead 1 \
                    # --optimizer sgd \