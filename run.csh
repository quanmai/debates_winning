#!/bin/sh

CUDA_VISIBLE_DEVICES=5 python main.py \
                    --batch-size 128 \
                    --nhid 32 \
                    --patience 100 \
                    --accelerator gpu \
                    --epoch 200 \
                    --is-counter \
                    --is-support \
                    --lr 1e-04 \
                    --scheduler exp \
                    --loss pair \
                    --v1 \
                    --embedding first \
                    --node-encoder-direction out \
                    --res \
                    --test-ver 631 \
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