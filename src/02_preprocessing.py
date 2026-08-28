# %% [markdown]
# # Milestone 1: Data Preprocessing & Windowing
# This script processes the MIT-BIH dataset into fixed-size heartbeat windows
# using the AAMI patient-aware training/testing split.

# %%
import wfdb
import numpy as np
import os
import collections
import urllib.request

# Project root (parent of this script's directory) so paths resolve from any cwd.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# %% [markdown]
# ### 1. Define the AAMI Patient Split and Labels
# To prevent data leakage, we divide patients into DS1 (Train) and DS2 (Test).

# %%
# DS1: Used for training the model
DS1_TRAIN = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', 
             '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', 
             '223', '230']

# DS2: Used strictly for evaluating the model
DS2_TEST = ['100', '103', '104', '105', '111', '113', '117', '121', '123', '200', 
            '202', '210', '212', '213', '214', '217', '219', '221', '222', '228', 
            '231', '232', '233', '234']

# Label Mapping for Binary Classification
NORMAL_CLASSES = ['N', 'L', 'R', 'e', 'j']
ABNORMAL_CLASSES = ['V', 'E', 'A', 'a', 'J', 'S', 'F']

# Windowing parameters
FS = 360
WINDOW_BEFORE = 90  # 0.25 seconds before R-peak
WINDOW_AFTER = 90   # 0.25 seconds after R-peak
WINDOW_SIZE = WINDOW_BEFORE + WINDOW_AFTER

# %% [markdown]
# ### 2. The Extraction Function
# This function loads a patient record, finds all valid R-peaks, slices the 180-sample 
# window around the peak, and normalizes it.

# %%
def extract_windows(records, data_dir):
    X, y = [], []
    class_counts = collections.Counter()
    
    print(f"Processing {len(records)} records...")
    
    for record_name in records:
        record_path = os.path.join(data_dir, record_name)
        
        # Download record if it doesn't exist locally
        if not os.path.exists(record_path + '.dat'):
            wfdb.dl_database('mitdb', dl_dir=data_dir, records=[record_name])
            
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        
        # We use channel 0 (usually MLII)
        signal = record.p_signal[:, 0] 
        
        for i, symbol in enumerate(annotation.symbol):
            # 1. Check if the symbol is one of our target classes
            if symbol in NORMAL_CLASSES:
                label = 0
            elif symbol in ABNORMAL_CLASSES:
                label = 1
            else:
                continue # Ignore unclassified or artifact beats
                
            # 2. Get the R-peak index
            peak_idx = annotation.sample[i]
            
            # 3. Ensure the window doesn't go out of bounds of the signal
            if peak_idx - WINDOW_BEFORE >= 0 and peak_idx + WINDOW_AFTER < len(signal):
                
                # Slice the array
                window = signal[peak_idx - WINDOW_BEFORE : peak_idx + WINDOW_AFTER]
                
                # Z-score Normalization (Mean=0, Std=1)
                # This removes baseline wander and scales the voltage
                mean_val = np.mean(window)
                std_val = np.std(window)
                if std_val > 0:
                    window = (window - mean_val) / std_val
                else:
                    window = window - mean_val
                    
                X.append(window)
                y.append(label)
                class_counts[label] += 1
                
    return np.array(X), np.array(y), class_counts

# %% [markdown]
# ### 3. Execute Extraction and Save to Disk
# We will run the extraction for both DS1 and DS2, reshaping the data 
# to fit a 1D CNN (Samples, TimeSteps, Channels).

# %%
data_dir = os.path.join(PROJECT_ROOT, 'data', 'mitdb')
os.makedirs(data_dir, exist_ok=True)
processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
os.makedirs(processed_dir, exist_ok=True)

print("--- Extracting Training Data (DS1) ---")
X_train, y_train, train_counts = extract_windows(DS1_TRAIN, data_dir)

print("\n--- Extracting Testing Data (DS2) ---")
X_test, y_test, test_counts = extract_windows(DS2_TEST, data_dir)

# Reshape for 1D CNN: (Samples, 180, 1)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print("\n=== PREPROCESSING COMPLETE ===")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"Train Class Balance: Normal={train_counts[0]}, Abnormal={train_counts[1]}")

print(f"\nX_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Test Class Balance: Normal={test_counts[0]}, Abnormal={test_counts[1]}")

# Save the processed numpy arrays
np.save(os.path.join(processed_dir, 'X_train.npy'), X_train)
np.save(os.path.join(processed_dir, 'y_train.npy'), y_train)
np.save(os.path.join(processed_dir, 'X_test.npy'), X_test)
np.save(os.path.join(processed_dir, 'y_test.npy'), y_test)
print(f"\nArrays saved successfully to {processed_dir}/")
