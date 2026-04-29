#!/bin/bash
# Demo: training the same model under different federated aggregation methods
# on the public PLAID benchmark.
#
# Run from this directory after generating client splits:
#   python preprocess.py --num_clients 100
#
# Each command below trains one method end-to-end. Results land under
# ./checkpoints/<experiment_name>/.

# FedAvg
python main.py \
    --aggregation fedavg \
    --num_clients 100 --client_fraction 0.1 \
    --num_rounds 600 --local_epochs 2 \
    --model cnn_medium --lr 0.005 --batch_size 128

# FedProx (proximal term mu=0.01)
python main.py \
    --aggregation fedprox --fedprox_mu 0.01 \
    --num_clients 100 --client_fraction 0.1 \
    --num_rounds 600 --local_epochs 2 \
    --model cnn_medium --lr 0.005 --batch_size 128

# PerFedNILM (personalised federated NILM)
python main.py \
    --aggregation perfednilm --fedprox_mu 0.01 \
    --num_clients 100 --client_fraction 0.1 \
    --num_rounds 600 --local_epochs 2 \
    --model cnn_medium --lr 0.005 --batch_size 128

# OLFED — online federated knowledge distillation with proximal regularisation
# (this paper). Teacher LR and KD start round are the two extra knobs.
python main.py \
    --aggregation fedkd_prox --fedprox_mu 0.01 \
    --Tlr 0.005 --kd_start_round 150 \
    --num_clients 100 --client_fraction 0.1 \
    --num_rounds 600 --local_epochs 2 \
    --model cnn_medium --lr 0.005 --batch_size 128
