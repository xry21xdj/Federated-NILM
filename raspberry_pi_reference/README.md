# Raspberry Pi Federated Learning Reference

A Raspberry-Pi-deployable federated learning reference implementation that
mirrors the federated training and edge-side communication used in OLFED.
The release uses the public **EMNIST** benchmark as a stand-in dataset so
that researchers without specialised NILM hardware can verify and adapt the
edge-deployment behaviour on commodity Pi devices.

The implementation is MQTT-based: clients and the server exchange model
parameters over a lightweight pub/sub channel. The directory is split into
two self-contained halves (`client/` and `server/`) that can be deployed on
separate machines.

## Setup

On every machine that will participate (server + each Pi client):

```bash
pip install paho-mqtt torch torchvision tqdm numpy pillow
```

Run an MQTT broker reachable by every node, e.g. `mosquitto`:

```bash
sudo apt-get install mosquitto
sudo systemctl start mosquitto
```

In `client/client.py` and `server/server.py`, set `broker_address` to the IP
or hostname of the machine running the broker (defaults to `"localhost"`,
which works when the broker is co-located with `server.py`).

## Running

On the **server** machine:

```bash
cd server/
python server.py --experiment emnist --method FedAvg --n_rounds 8 \
    --bz 128 --lr 0.01 --local_steps 1 --optimizer sgd --seed 1234
```

On each **client** machine (a Raspberry Pi or any host with Python and
network access to the broker):

```bash
cd client/
python client.py
```

`server/run_server.sh` lists ready-to-run commands for the included
aggregation methods (FedAvg, FedMom, FedAdam, FedMGDA, FedNAG, FastSlowMo,
DOMO, MIME, FedMoS, FedGLOMO, FedAvg-M, FedMGDA-M-Mom, FedMGDA-Mom).
`client/run_client.sh` launches multiple `client.py` processes locally for
single-machine testing.

## File layout

```
raspberry_pi_reference/
├── client/   # client-side: data loaders, local optim, MQTT subscriber
└── server/   # server-side: aggregation rules, weight calc, MQTT publisher
```

Each side carries its own `args.py`, `model.py`, `data_handle.py`, and
`optim.py`, so the two halves can be deployed independently. To swap in a
different dataset, replace the data loader in `client/data_handle.py` and
`server/data.py` and re-point `--experiment` accordingly.

## License

MIT — see `LICENSE`.
