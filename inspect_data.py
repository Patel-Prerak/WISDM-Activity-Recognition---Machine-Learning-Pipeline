import pandas as pd
import numpy as np

columns = ['user', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']

def load_data(file_path):
    data = []
    line_count = 0
    with open(file_path, 'r') as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if not line:
                continue
            if line.endswith(';'):
                line = line[:-1]
            
            parts = [p.strip() for p in line.split(',')]
            
            if len(parts) == 6:
                try:
                    # Check if all numerical parts are actually numerical
                    # parts[0] is user (int)
                    # parts[2] is timestamp (int)
                    # parts[3], parts[4], parts[5] are floats
                    user = int(parts[0])
                    activity = parts[1]
                    timestamp = int(float(parts[2])) # some might be scientific notation or float strings
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])
                    data.append([user, activity, timestamp, x, y, z])
                except ValueError:
                    # print(f"Skipping line {line_count}: {line}")
                    continue
            else:
                # print(f"Skipping line {line_count}: {line}")
                continue
    
    df = pd.DataFrame(data, columns=columns)
    return df

file_path = 'WISDM_ar_v1.1_raw.txt'
df = load_data(file_path)

if df is not None:
    print("Data loaded successfully.")
    print(f"Shape: {df.shape}")
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nActivity counts:")
    print(df['activity'].value_counts())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for unique users
    print(f"\nUnique users: {df['user'].nunique()}")
