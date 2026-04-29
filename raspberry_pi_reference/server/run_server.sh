python server.py --experiment emnist --method FedAvg --n_rounds 8 --bz 128 --lr 0.01 --local_steps 1 --gamma 1.0 --beta 0.0 --mu 0.0 --log_freq 10 --optimizer sgd --seed 1234

python server.py --experiment emnist --method FedMGDA --n_rounds 8 --bz 128 --lr 0.01 --local_steps 1 --gamma 1.0 --beta 0.0 --mu 0.0 --log_freq 10 --optimizer sgd --seed 1234

python server.py --experiment emnist --method FedNAG --n_rounds 8 --bz 128 --lr 0.01 --local_steps 1 --gamma 1.0 --beta 0.0 --mu 0.0 --log_freq 10 --optimizer nag_sgd --seed 1234

python server.py --experiment emnist --method FedMom --n_rounds 8 --bz 128 --lr 0.01 --local_steps 1 --gamma 1.0 --beta 0.0 --mu 0.0 --log_freq 10 --optimizer nag_sgd --seed 1234

python server.py --experiment emnist --method FastSlowMo --n_rounds 8 --bz 128 --lr 0.01 --local_steps 1 --gamma 0.5 --beta 0.5 --mu 0. --log_freq 10 --optimizer fastslowmo_sgd --seed 1234

python server.py --experiment emnist --method MIME --n_rounds 8 --bz 128 --lr 0.01 --local_steps 1 --gamma 1.0 --beta 0.1 --mu 0. --log_freq 10 --optimizer mime_sgd --seed 1234

python server.py --experiment emnist --method FedMoS --n_rounds 8 --bz 128 --lr 0.01 --local_steps 1 --gamma 0.9 --beta 0.5 --mu 0. --log_freq 10 --optimizer fedmos_sgd --seed 1234

python server.py --experiment emnist --method DOMO --n_rounds 8 --bz 128 --lr 0.01 --local_steps 1 --gamma 1.0 --beta 0.5 --mu 0.5 --log_freq 10 --optimizer domo_sgd --seed 1234

python server.py --experiment emnist --method FedGLOMO --n_rounds 8 --bz 128 --lr 0.01 --local_steps 1 --gamma 1.0 --beta 0.9 --mu 0. --log_freq 10 --optimizer fedglomo_sgd --seed 1234

python server.py --experiment emnist --method FedAvg-M --n_rounds 8 --bz 128 --lr 0.01 --local_steps 1 --gamma 1.0 --beta 0.9 --mu 0. --log_freq 10 --optimizer m_sgd --seed 1234

python server.py --experiment emnist --method FedAvg-M-Mom --n_rounds 8 --bz 128 --lr 0.01 --local_steps 1 --gamma 1.0 --beta 0.5 --mu 0.5 --log_freq 10 --optimizer m_sgd --seed 1234

python server.py --experiment emnist --method FedMGDA-M-Mom --n_rounds 8 --bz 128 --lr 0.01 --local_steps 1 --gamma 1.0 --beta 0.5 --mu 0.5 --log_freq 10 --optimizer m_sgd --seed 1234

python server.py --experiment emnist --method FedMGDA-Mom --n_rounds 8 --bz 128 --lr 0.01 --local_steps 1 --gamma 1.0 --beta 0.5 --mu 0.5 --log_freq 10 --optimizer sgd --seed 1234



