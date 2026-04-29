# FL-NILM (OLFED)

Federated learning for non-intrusive load monitoring (NILM) with optional
differential privacy. Companion code release for the OLFED paper.

This repository is organised into two complementary parts:

| Directory | What it provides |
|---|---|
| [`plaid_experiments/`](plaid_experiments/) | Full training pipeline for OLFED and federated baselines (FedAvg, FedProx, PerFedNILM) on the public PLAID benchmark, with optional differential privacy. Demos cover aggregation methods, client-fraction scalability, and DP. |
| [`raspberry_pi_reference/`](raspberry_pi_reference/) | Raspberry-Pi-deployable federated learning reference using the public EMNIST benchmark, MQTT-based client/server communication, and a suite of FL aggregation methods. Verifies the edge-side behaviour underpinning OLFED on commodity Pi hardware. |

See each subdirectory's `README.md` for setup, dataset placement, and
runnable demos.

## License

MIT — see `LICENSE`.
