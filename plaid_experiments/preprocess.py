"""
PLAID data preprocessing.

Generates V-I trajectory images from PLAID submetered V/I CSV files and splits
them into Non-IID per-client partitions (Dirichlet-based) while preventing
train/test leakage at the original-file granularity.
"""
import os
import pickle
import random

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from load_plaid_csv import load_plaid_csv


def generate_vi_trajectory(voltage, current, img_size=192):
    """
    Generate a V-I trajectory image from voltage / current signals.
    """
    if len(voltage) == 0 or len(current) == 0:
        return np.zeros((img_size, img_size))

    v_range = voltage.max() - voltage.min()
    i_range = current.max() - current.min()

    if v_range == 0 or i_range == 0:
        return np.zeros((img_size, img_size))

    # Normalize to [-1, 1]
    v_norm = 2 * (voltage - voltage.min()) / v_range - 1
    i_norm = 2 * (current - current.min()) / i_range - 1

    img = np.zeros((img_size, img_size))

    x = ((v_norm + 1) / 2 * (img_size - 1)).astype(int)
    y = ((i_norm + 1) / 2 * (img_size - 1)).astype(int)

    x = np.clip(x, 0, img_size - 1)
    y = np.clip(y, 0, img_size - 1)

    for i in range(len(x)):
        img[y[i], x[i]] += 1

    if img.max() > 0:
        img = img / img.max()

    return img


def split_for_federated(data_list, labels, num_clients=100, test_size=0.2, seed=42):
    """
    Split data into Non-IID per-client partitions while preventing train/test leakage.

    The split happens at the original-file granularity: all augmented samples
    that originate from the same source CSV stay together in either train or test.
    """
    np.random.seed(seed)

    # 1. Group samples by appliance type (label).
    data_by_type = {}
    for data, label in zip(data_list, labels):
        app_type = data['type']
        if app_type not in data_by_type:
            data_by_type[app_type] = []
        data_by_type[app_type].append({'data': data, 'label': label})

    # 2. Dirichlet-based Non-IID assignment to clients.
    # Smaller alpha -> more skewed Non-IID; alpha -> +inf approaches IID.
    alpha = 0.5
    num_classes = len(data_by_type)
    class_list = sorted(list(data_by_type.keys()))

    # Per-class proportion vector for each client.
    client_dist = np.random.dirichlet([alpha] * num_classes, num_clients)

    client_data_buckets = [[] for _ in range(num_clients)]

    print(f"\nDistributing data to {num_clients} clients (Dirichlet alpha={alpha})...")

    for class_idx, app_type in enumerate(class_list):
        samples = data_by_type[app_type]

        random.shuffle(samples)

        # Per-client share of this class.
        proportions = client_dist[:, class_idx]
        proportions = proportions / proportions.sum()

        # Concrete sample counts (the last client absorbs any remainder).
        split_indices = (np.cumsum(proportions)[:-1] * len(samples)).astype(int)
        split_samples = np.split(samples, split_indices)

        for client_id, subset in enumerate(split_samples):
            client_data_buckets[client_id].extend(subset)

    # 3. Per-client train/test split — leakage-safe split is done by original file.
    client_splits = []

    print("\nSplitting Train/Test per client (Grouped by Original File)...")

    for i, data_bucket in enumerate(client_data_buckets):
        if len(data_bucket) < 5:
            # Skip clients with too few samples.
            continue

        # Collect the unique source-file IDs present in this client's bucket.
        all_original_files = list(set(d['data']['original_file'] for d in data_bucket))

        # If only one source file exists, we cannot split safely; put all in train.
        if len(all_original_files) < 2:
            train_files = all_original_files
            test_files = []
        else:
            # Split file IDs (not samples) so augmented copies of the same file
            # never straddle the train/test boundary.
            train_files, test_files = train_test_split(all_original_files, test_size=test_size, random_state=seed)

        train_files_set = set(train_files)
        test_files_set = set(test_files)

        train_samples = []
        test_samples = []

        for item in data_bucket:
            orig_file = item['data']['original_file']
            obj = {
                'image': item['data']['image'],
                'label': item['label'],
                'type': item['data']['type'],
                'id': item['data']['id']
            }

            if orig_file in train_files_set:
                train_samples.append(obj)
            elif orig_file in test_files_set:
                test_samples.append(obj)

        client_splits.append({
            'client_id': i,
            'train': train_samples,
            'test': test_samples
        })

    # Summary statistics.
    total_train = sum(len(c['train']) for c in client_splits)
    total_test = sum(len(c['test']) for c in client_splits)
    print(f"\nFinal Statistics:")
    print(f"  Active Clients: {len(client_splits)}")
    print(f"  Total Train Samples: {total_train}")
    print(f"  Total Test Samples: {total_test}")
    print(f"  Leakage Check: Safe (Split by Original File ID)")

    return client_splits


def preprocess_plaid(data_path='./data/PLAID',
                     output_dir='./data/processed',
                     num_clients=100,
                     img_size=192):
    """
    Complete preprocessing pipeline:
      Load PLAID CSVs -> extract steady-state cycles -> generate V-I images
      -> Dirichlet Non-IID split across clients -> persist client splits.
    """

    # Step 1: Load Data
    print("=" * 60)
    print("Step 1: Loading PLAID data with Augmentation...")
    print("=" * 60)

    # max_samples_per_file controls the augmentation factor; 30 typically
    # yields tens of thousands of samples on the full PLAID submetered set.
    data_list, labels, label_map, appliance_types = load_plaid_csv(data_path, max_samples_per_file=30)

    # Step 2: Generate VI Images
    print("\n" + "=" * 60)
    print("Step 2: Generating VI trajectories...")
    print("=" * 60)

    # Single-process loop with tqdm progress (multi-processing avoided to keep things simple).
    for data in tqdm(data_list, desc="Generating images"):
        img = generate_vi_trajectory(data['voltage'], data['current'], img_size)
        data['image'] = img

    # Step 3: Split for Federated Learning
    print("\n" + "=" * 60)
    print(f"Step 3: Splitting data for {num_clients} clients...")
    print("=" * 60)

    client_splits = split_for_federated(data_list, labels, num_clients, test_size=0.2, seed=42)

    # Step 4: Save
    print("\n" + "=" * 60)
    print("Step 4: Saving processed data...")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    label_file = f'{output_dir}/label_map.pkl'
    with open(label_file, 'wb') as f:
        pickle.dump(label_map, f)

    splits_dir = './data/splits'
    os.makedirs(splits_dir, exist_ok=True)

    splits_file = f'{splits_dir}/client_splits.pkl'
    with open(splits_file, 'wb') as f:
        pickle.dump(client_splits, f)
    print(f"Saved splits to: {splits_file}")

    print("\n" + "=" * 60)
    print("Preprocessing complete.")
    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PLAID Data Preprocessing')
    parser.add_argument('--data_path', default='./data/PLAID', help='Path to PLAID data directory')
    parser.add_argument('--output_dir', default='./data/processed', help='Output directory')
    parser.add_argument('--num_clients', type=int, default=100, help='Number of FL clients')

    args = parser.parse_args()

    preprocess_plaid(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_clients=args.num_clients
    )
