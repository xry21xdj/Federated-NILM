# PLAID Experiments — OLFED and Federated Baselines

Reference training pipeline for OLFED and the federated baselines (FedAvg,
FedProx, PerFedNILM) on the public PLAID NILM benchmark, with optional
Differential Privacy.

## Setup

```bash
pip install -r requirements.txt
```

PLAID is freely available from
<https://figshare.com/articles/dataset/PLAID_-_A_Voltage_and_Current_Measurement_Dataset_for_Plug_Load_Appliance_Identification_in_Households/10084619?file=18183113>
(Medico et al., *Sci. Data* 2020). Download the submetered recordings and
`metadata_submetered.json`, then unpack so that the per-appliance CSVs and
the metadata file sit directly under `./data/PLAID/`:

```
./data/PLAID/
├── metadata_submetered.json
├── 1.csv
├── 2.csv
└── ...
```

## Quick start

```bash
# (1) Generate per-client splits once.
python preprocess.py --num_clients 100

# (2) Train one method (e.g., OLFED) end-to-end.
python main.py \
    --aggregation fedkd_prox --fedprox_mu 0.01 \
    --Tlr 0.005 --kd_start_round 150 \
    --num_clients 100 --client_fraction 0.1 \
    --num_rounds 600 --local_epochs 2 \
    --model cnn_medium --lr 0.005 --batch_size 128
```

Checkpoints are written to `./checkpoints/<experiment_name>/global_model.pt`.
Pass `--use_wandb` to additionally log to Weights & Biases.

## Demos

| Script | What it shows |
|---|---|
| `demo_aggregation_methods.sh` | Same model trained under FedAvg, FedProx, PerFedNILM, and OLFED |
| `demo_client_fraction.sh`     | Scalability sweep: 4 / 10 / 20 / 50 active clients |
| `demo_differential_privacy.sh`| Single DP run + privacy-utility sweep ($\epsilon \in [20, 50]$) |

Each demo is self-contained and runnable from this directory. Edit the
hyper-parameters at the top of each file to suit your hardware.

## File layout

```
plaid_experiments/
├── main.py              # entry point (training / evaluation)
├── client.py            # FL client (local training, KD, evaluation)
├── server.py            # FL server (aggregation, DP, logging)
├── aggregators.py       # FedAvg / FedProx / PerFedNILM / OLFED aggregation
├── models.py            # CNN architectures (cnn / cnn_medium / cnn_small / etc.)
├── preprocess.py        # PLAID -> V-I trajectory image (PE4IP encoding)
├── load_plaid_csv.py    # PLAID raw CSV loader
├── config.py            # default hyper-parameters
├── utils.py             # checkpointing, seeds, device selection
└── demo_*.sh            # runnable demos
```

## Differential Privacy

Per-client gradient clipping + Gaussian noise on the aggregated update. The
noise sigma is derived from $(\epsilon, \delta)$ via the analytic Gaussian
mechanism (see `Config.calculate_dp_noise_scale`). Enable with `--use_dp`
and configure `--dp_epsilon`, `--dp_delta`, `--dp_clip_norm`.

## License

MIT — see `LICENSE`.
