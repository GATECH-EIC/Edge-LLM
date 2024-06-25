import os
import numpy as np
import pandas as pd

def min_max_scaling(data):
    min_val = min(data)
    max_val = max(data)
    scaled_data = [(x - min_val) / (max_val - min_val) for x in data]
    return scaled_data
    
def schedular(data, total_pruning_ratio):
    inverted_data = [1 - x for x in data]

    # Softmax transformation with temperature
    def softmax_with_temperature(x, T):
        e_x = np.exp((x - np.max(x)) / T)  # subtract max(x) for numerical stability
        return e_x / e_x.sum()

    # Choose a temperature less than 1 to amplify differences
    T = 0.1
    softmaxed_data = softmax_with_temperature(inverted_data, T)

    # Scaling to make the sum 0.2
    scaled_data = [x * total_pruning_ratio for x in softmaxed_data]

    return inverted_data

def get_pruning_schedular(type, file_path, total_pruning_ratio=0.5, layer_num=32):
    total_pruning_ratio = total_pruning_ratio * layer_num
    pruning_ratios =  ['0.05', '0.1', '0.2', '0.3', '0.5',  '0.7',  '0.9']

    files = []
    for root, dirs, filenames in os.walk(file_path):
        for filename in filenames:
            files.append(os.path.join(root, filename))

    all_MSE_results = list()
    for file in files:
        MSE_results = pd.read_feather(file) 
        results = list()
        for row in MSE_results.itertuples():
            temp_list = list()
            for i in range(1, 8):
                if type == "ordinary":
                    temp_list.append(float(row[i]))
                elif type == "sqrt":
                    temp_list.append(np.sqrt(float(row[i])))
                elif type == "square":
                    temp_list.append(float(row[i]) ** 2)
            results.append(np.mean(temp_list))
        all_MSE_results.append(results)   
    
    bit_layer_results = list()
    for i in range(len(pruning_ratios)):
        temp_list = list()
        for data in all_MSE_results:
            temp_list.append(data[i])
        bit_layer_results.append(temp_list)

    mean_MSE_layerwise = np.mean(bit_layer_results, axis=0)
    scaled_data = min_max_scaling(mean_MSE_layerwise)

    scaled_data = [1-item for item in scaled_data]
    ratio = [item / sum(scaled_data) * total_pruning_ratio for item in scaled_data]

    return ratio
