import pywt
import numpy as np
import pandas as pd




def getPacketsSizes(data):
    packets_sizes = []
    for i in range(0, 30): # get the first 30 packets sizes from the csv data
        packet_size = data[f"first_packet_sizes_{i}"]
        packets_sizes.append(packet_size)
    return packets_sizes



def wavelet(packet_sizes):
    wavelet_type='sym6'
    coeffs = pywt.wavedec(packet_sizes, wavelet_type)
    # Extract the approximation coefficients (cA) and detail coefficients (cD)
    cA = coeffs[0]  # Approximation coefficients
    # Calculate wavelet leaders
    wavelet_leaders = np.abs(cA) * np.sqrt(np.arange(1, len(cA) + 1))
    wavelet_leaders = [round(x, 3) for x in wavelet_leaders]
    # if len(wavelet_leaders) < 20:
    for i in range(len(wavelet_leaders),20):
        wavelet_leaders.append(0)
    return wavelet_leaders


def extract_features(path: str, features_type: str):
    # Define the required features based on the image and your description
    selected_features = []
    
    if features_type == 'BOA':
        # Add the first packet sizes 9,21,24,28
        selected_features.extend(["first_packet_sizes_9", "first_packet_sizes_21", "first_packet_sizes_24", "first_packet_sizes_28"])
        # Add beacon 10,11,15
        selected_features.extend(["beaconning_10", "beaconning_11", "beaconning_15"])
        # Add dst2src mean piat
        selected_features.extend(["mean_bwd_inter_arrival_time"])
        # Add wavelet leaders 0,16,7
        selected_features.extend(["wavelet_0", "wavelet_16", "wavelet_7"])
        # Add the label
        selected_features.append("label")
        
    else: # not functional yet
        pass
        
    # Read the dataset
    data = pd.read_csv(path)
    
    # For each flow, calculate the wavelet leaders of the first 30 packets sizes
    wavelet_leaders = []
    for index, row in data.iterrows():
        packets_sizes = getPacketsSizes(row)
        wavelet_leaders.append(wavelet(packets_sizes))
    
    # Add the wavelet leaders to the filtered dataset, only wavelet {0,16,7}
    wavelet_leaders_df = pd.DataFrame(wavelet_leaders, columns=["wavelet_0", "wavelet_1", "wavelet_2", "wavelet_3", "wavelet_4", "wavelet_5", "wavelet_6", "wavelet_7", "wavelet_8", "wavelet_9", "wavelet_10", "wavelet_11", "wavelet_12", "wavelet_13", "wavelet_14", "wavelet_15", "wavelet_16", "wavelet_17", "wavelet_18", "wavelet_19"])
    filtered_data = pd.concat([data, wavelet_leaders_df], axis=1)
    # Filter the dataset to include only the selected features
    filtered_data = filtered_data[selected_features]
    
    output_file = path.split(".")[0] + "_features.csv"
    
    # Save the filtered dataset to a new csv file
    filtered_data.to_csv(output_file, index=False)

# load the dataset

APP_PATH = r'C:\Users\tnrha\Attack-Detection-Project\datasets\Tournament\app_complete.csv'
ATTRIBUTION_PATH = r'C:\Users\tnrha\Attack-Detection-Project\datasets\Tournament\attribution_complete.csv'
extract_features(APP_PATH,"BOA")
extract_features(ATTRIBUTION_PATH,"BOA")

