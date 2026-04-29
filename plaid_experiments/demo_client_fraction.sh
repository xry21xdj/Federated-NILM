#!/bin/bash
# Demo: scalability — sweeping the number of active clients per round.
#
# We use a single 100-client preprocessed split (generated once via
# `python preprocess.py --num_clients 100`) and select a fraction per round.
#   fraction 0.04 = 4 active clients   (matches the main paper experiment)
#   fraction 0.10 = 10 active clients
#   fraction 0.20 = 20 active clients
#   fraction 0.50 = 50 active clients
# Other knobs are held fixed so the only varying axis is client count.

for FRACTION in 0.04 0.10 0.20 0.50; do
    python main.py \
        --aggregation fedavg \
        --num_clients 100 --client_fraction ${FRACTION} \
        --num_rounds 600 --local_epochs 2 \
        --model cnn_medium --lr 0.005 --batch_size 128
done
