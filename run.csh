#!/bin/sh

CUDA_VISIBLE_DEVICES=6 python main.py \
                    --nhid 32 \
                    --patience 5 \
                    --epoch 100 \
                    --lr 1e-04 \
                    --batch-size 4 \
                    --nhead 4 \
                    --loss pair \
                    --v1 \
                    --accelerator gpu \
                    --check-val-freq 1 \
                    --is-counter \
                    # --user-emb \
                    # --turn-emb \
                    # --pos-emb \
                    # --sparsify threshold \
                    # --nogat \
                    # --mode bidirection \
                    # --gat-layers 2 \
                    # --counter-coeff 0.8 \
                    # --nhead 1 \
                    # --optimizer sgd \