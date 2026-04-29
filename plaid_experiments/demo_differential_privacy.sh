#!/bin/bash
# Demo: enabling Differential Privacy (DP-SGD style noise + per-client gradient
# clipping) during federated training.
#
# The DP machinery is wired into the standard pipeline; turn it on with
# `--use_dp` and configure the (epsilon, delta, clip-norm) triplet.
# At each round, per-client updates are clipped to L2 norm `dp_clip_norm`
# and Gaussian noise is added to the aggregated update with sigma derived
# from (epsilon, delta) via the standard analytic Gaussian mechanism
# (see Config.calculate_dp_noise_scale in config.py).

# (1) Single DP run at moderate privacy budget (eps=30, delta=1e-2, C=1.0).
python main.py \
    --aggregation fedkd_prox --fedprox_mu 0.01 \
    --Tlr 0.005 --kd_start_round 150 \
    --num_clients 100 --client_fraction 0.04 \
    --num_rounds 600 --local_epochs 2 \
    --model cnn_medium --lr 0.005 --batch_size 128 \
    --use_dp --dp_epsilon 30 --dp_delta 1e-2 --dp_clip_norm 1.0

# (2) Privacy-utility sweep (epsilon = 20..50). Other DP params held fixed.
for EPS in 20 25 30 35 40 45 50; do
    python main.py \
        --aggregation fedkd_prox --fedprox_mu 0.01 \
        --Tlr 0.005 --kd_start_round 150 \
        --num_clients 100 --client_fraction 0.04 \
        --num_rounds 600 --local_epochs 2 \
        --model cnn_medium --lr 0.005 --batch_size 128 \
        --use_dp --dp_epsilon ${EPS} --dp_delta 1e-2 --dp_clip_norm 1.0
done
