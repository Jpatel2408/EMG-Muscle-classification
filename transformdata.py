import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import h5py

# Example paths
base_dir = "data"
participants = [f"subject {i}" for i in range(1, 24)]
muscle_groups = ["biceps-isom", "biceps-isot", "triceps-isot", "triceps-isom", "Gastrocnemius-Medialis-isom", "Gastrocnemius-Medialis-isot", "Vastus-Medialis-isom", "Vastus-Medialis-isot"]

def load_data(file_path):
    # Example to load data from a file (adjust according to your file type)
    with h5py.File(file_path, 'r') as hdf:
        # Assuming 'raw/channel_2' contains the EMG data
        data = np.array(hdf['00:07:80:8C:06:8B/raw/channel_2']).flatten()
    return data

def bandpass_filter(data, lowcut=20, highcut=500, fs=1000, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if high >= 1.0:
        high = 0.99

    # Debugging statement
    print(f"Lowcut (normalized): {low}, Highcut (normalized): {high}")
    
    if low <= 0 or high <= 0 or low >= high:
        raise ValueError(f"Invalid cutoff frequencies: low = {low}, high = {high}.")
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def extract_features(data):
    # Extract features from filtered EMG data
    features = {
        'rms': np.sqrt(np.mean(data**2)),
        'mav': np.mean(np.abs(data)),
        'zc': np.sum(np.diff(np.sign(data)) != 0),
        'ssc': np.sum(np.diff(np.sign(np.diff(data))) != 0),
        'wavelength': np.sum(np.abs(np.diff(data)))
    }
    return features

# DataFrame to hold all data
all_data = []

for participant in participants:
    for muscle in muscle_groups:
        file_path = os.path.join(base_dir, participant, f"{muscle}.h5")  # Adjust file format
        raw_data = load_data(file_path)
        filtered_data = bandpass_filter(raw_data)
        features = extract_features(filtered_data)
        features['participant'] = participant
        features['muscle_group'] = muscle
        all_data.append(features)

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Save processed data
df.to_csv("processed_emg_data.csv", index=False)

# Now df can be used for machine learning tasks
