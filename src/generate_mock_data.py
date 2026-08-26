import numpy as np
import collections
import os

def generate_mock_data():
    os.makedirs('../data/processed', exist_ok=True)
    
    # Generate DS1 (Train)
    num_train = 50000
    # 90% Normal (0), 10% Abnormal (1)
    y_train = np.random.choice([0, 1], size=num_train, p=[0.9, 0.1])
    X_train = np.random.randn(num_train, 180, 1).astype(np.float32)
    # Add a slight pattern so the model can actually "learn" something and not just guess
    for i in range(num_train):
        if y_train[i] == 1:
            # Add an anomaly spike in the middle
            X_train[i, 80:100, 0] += 5.0
            
    # Generate DS2 (Test)
    num_test = 20000
    y_test = np.random.choice([0, 1], size=num_test, p=[0.9, 0.1])
    X_test = np.random.randn(num_test, 180, 1).astype(np.float32)
    for i in range(num_test):
        if y_test[i] == 1:
            X_test[i, 80:100, 0] += 5.0

    print(f"Generated X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Generated X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    np.save('../data/processed/X_train.npy', X_train)
    np.save('../data/processed/y_train.npy', y_train)
    np.save('../data/processed/X_test.npy', X_test)
    np.save('../data/processed/y_test.npy', y_test)

if __name__ == '__main__':
    generate_mock_data()
