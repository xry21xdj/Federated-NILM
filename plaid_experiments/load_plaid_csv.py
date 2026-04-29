"""
PLAID CSV-format data loader with sliding-window augmentation.

Loads the PLAID submetered dataset and extracts multiple steady-state cycles
per source file. PLAID is a public dataset of high-frequency current/voltage
measurements of household appliances:

    Medico, R., De Baets, L., Gao, J. et al. "A voltage and current measurement
    dataset for plug load appliance identification in households."
    Sci Data 7, 49 (2020). https://doi.org/10.1038/s41597-020-0389-7

Download / project page:
    https://figshare.com/articles/dataset/PLAID_-_A_Voltage_and_Current_Measurement_Dataset_for_Plug_Load_Appliance_Identification_in_Households/10084619
"""
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

def extract_steady_state_cycle(voltage, current, num_cycles=2, stride_cycles=1, max_samples=30):
    """
    Extract multiple steady-state cycles using a sliding window approach.
    
    Args:
        voltage: Voltage array
        current: Current array
        num_cycles: Number of cycles per sample (default 2)
        stride_cycles: How many cycles to move the window back (default 1 for overlap)
        max_samples: Maximum number of samples to extract from one file
        
    Returns:
        v_list, i_list: Lists of extracted signal segments
    """
    # 1. Zero-Crossing Detection on Voltage
    zero_crossings = np.where(np.diff(np.sign(voltage)))[0]
    
    # Calculate required crossings
    # 1 cycle = 2 crossings. We need start and end points.
    window_crossings = num_cycles * 2
    stride_crossings = stride_cycles * 2
    
    # Check if file is long enough for at least one window
    if len(zero_crossings) < window_crossings + 2:
        return [], []
    
    extracted_v = []
    extracted_i = []
    
    # 2. Extract from the END (Backwards)
    # The end of the recording is usually the most stable steady state.
    # We use a pointer to the 'end' crossing of the current window.
    
    current_end_ptr = len(zero_crossings) - 1
    
    while len(extracted_v) < max_samples:
        # Check if we have enough history left for a full window
        if current_end_ptr - window_crossings < 0:
            break
            
        # Define window indices
        end_idx = zero_crossings[current_end_ptr]
        start_idx = zero_crossings[current_end_ptr - window_crossings]
        
        # Extract data
        v_seg = voltage[start_idx:end_idx]
        i_seg = current[start_idx:end_idx]
        
        # Basic integrity check (ensure not empty)
        if len(v_seg) > 0 and len(i_seg) > 0:
            extracted_v.append(v_seg)
            extracted_i.append(i_seg)
        
        # Move window backwards
        current_end_ptr -= stride_crossings
    
    return extracted_v, extracted_i


def load_plaid_csv(plaid_path, max_samples_per_file=30):
    """
    Load PLAID dataset with Data Augmentation
    
    Args:
        plaid_path: Path to PLAID data directory
        max_samples_per_file: Max augmented samples to generate per original CSV
    
    Returns:
        data_list, labels_list, label_map, appliance_types
    """
    
    print(f"Loading PLAID data from {plaid_path}")
    print(f"Augmentation Strategy: Max {max_samples_per_file} samples/file via sliding window")
    
    # Load metadata
    metadata_file = os.path.join(plaid_path, 'metadata_submetered.json')
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Build appliance type mapping
    appliance_types = set()
    for file_id, meta in metadata.items():
        app_type = meta.get('appliance', {}).get('type', 'unknown')
        if app_type and app_type != 'unknown':
            appliance_types.add(app_type)
    
    appliance_types = sorted(list(appliance_types))
    label_map = {app_type: idx for idx, app_type in enumerate(appliance_types)}
    
    print(f"\nFound {len(appliance_types)} appliance types")
    
    # Load CSV files
    data_list = []
    labels_list = []
    
    csv_dir = os.path.join(plaid_path, 'submetered')
    if not os.path.exists(csv_dir):
        csv_dir = plaid_path
    
    files_processed = 0
    total_samples_generated = 0
    skipped_files = 0
    
    print(f"\nLoading and augmenting data from {csv_dir}...")
    
    for file_id, meta in tqdm(metadata.items(), desc="Processing"):
        # Get appliance type
        app_type = meta.get('appliance', {}).get('type', None)
        if not app_type or app_type not in label_map:
            skipped_files += 1
            continue
        
        # Get CSV filename
        csv_file = os.path.join(csv_dir, f'{file_id}.csv')
        
        if not os.path.exists(csv_file):
            skipped_files += 1
            continue
        
        try:
            # Load CSV file
            df = pd.read_csv(csv_file, header=None)
            
            if len(df.columns) != 2:
                skipped_files += 1
                continue
            
            # Identify Voltage vs Current
            col0_mean = df.iloc[:, 0].abs().mean()
            col1_mean = df.iloc[:, 1].abs().mean()
            
            if col1_mean > col0_mean:
                current = df.iloc[:, 0].values
                voltage = df.iloc[:, 1].values
            else:
                current = df.iloc[:, 1].values
                voltage = df.iloc[:, 0].values
            
            # === KEY CHANGE: Extract List of Cycles (Augmentation) ===
            v_list, i_list = extract_steady_state_cycle(
                voltage, current, 
                num_cycles=2, 
                stride_cycles=1,   # 50% overlap for diversity
                max_samples=max_samples_per_file
            )
            
            if not v_list:
                skipped_files += 1
                continue
            
            # Process each extracted segment
            valid_segments = 0
            for idx, (v_seg, i_seg) in enumerate(zip(v_list, i_list)):
                
                # Validate Constraints (Paper: 110V <= V_RMS <= 130V, I_max <= 50A)
                v_rms = np.sqrt(np.mean(v_seg**2))
                i_max = np.max(np.abs(i_seg))
                
                # Loose constraints to allow for fluctuations
                if not (100 <= v_rms <= 140 and i_max <= 50):
                    continue
                
                # Create data dictionary
                # ID format: {original_file_id}_{segment_index}
                unique_id = f"{file_id}_{idx}"
                
                data_dict = {
                    'voltage': v_seg,
                    'current': i_seg,
                    'type': app_type,
                    'id': unique_id, 
                    'original_file': file_id, # Keep track of source
                    'location': meta.get('location', 'unknown'),
                    'v_rms': v_rms, 
                    'i_max': i_max
                }
                
                data_list.append(data_dict)
                labels_list.append(label_map[app_type])
                valid_segments += 1
            
            if valid_segments > 0:
                files_processed += 1
                total_samples_generated += valid_segments
            else:
                skipped_files += 1
            
        except Exception as e:
            # print(f"Error: {e}")
            skipped_files += 1
            continue
    
    print(f"\nProcessing Summary:")
    print(f"  Original Files Processed: {files_processed}")
    print(f"  Total Augmented Samples: {total_samples_generated}")
    print(f"Avg samples per file: {total_samples_generated/files_processed:.1f}")
    print(f"  Files Skipped: {skipped_files}")
    
    # Print distribution
    print("\nClass distribution (Augmented):")
    label_counts = Counter(labels_list)
    for app_type, idx in sorted(label_map.items(), key=lambda x: x[1]):
        count = label_counts[idx]
        print(f"  {app_type}: {count}")
    
    return data_list, labels_list, label_map, appliance_types


def validate_plaid_data(data_list, labels_list):
    """Validate loaded PLAID data"""
    print("\n" + "="*60)
    print("Data Validation (Augmented)")
    print("="*60)
    
    if len(data_list) == 0:
        print("No data loaded!")
        return

    print(f"Total samples: {len(data_list)}")
    
    # Check signal lengths
    voltage_lengths = [len(d['voltage']) for d in data_list]
    
    print(f"\nSignal lengths (samples):")
    print(f"  Min: {min(voltage_lengths)}")
    print(f"  Max: {max(voltage_lengths)}")
    print(f"  Mean: {np.mean(voltage_lengths):.0f}")
    
    # Check V_RMS distribution
    v_rms_list = [d['v_rms'] for d in data_list]
    print(f"\nV_RMS stats:")
    print(f"  Min: {min(v_rms_list):.2f} V")
    print(f"  Max: {max(v_rms_list):.2f} V")
    print(f"  Mean: {np.mean(v_rms_list):.2f} V")
    
    print("="*60)


if __name__ == '__main__':
    # Test loader
    path = './data/PLAID'
    if os.path.exists(path):
        # Setting max_samples to 50 for testing
        data_list, labels_list, label_map, appliance_types = load_plaid_csv(path, max_samples_per_file=50)
        validate_plaid_data(data_list, labels_list)
    else:
        print(f"Path {path} not found. Please check your data directory.")