import json
import torch
from safetensors.torch import safe_open

def total_size():
    # Load the index file to get the mapping of tensor names to safetensors files
    with open("/root/.cache/huggingface/hub/models--models--new_model/    /model.safetensors.index.json", "r") as file:
        index_data = json.load(file)
        weight_map = index_data["weight_map"]

    # Dictionary to hold the file paths that need to be opened
    files_to_open = set(weight_map.values())

    # Dictionary to collect tensors data
    tensors_info = {}

    # Open each safetensors file and read the necessary tensor data
    for safetensor_file in files_to_open:
        with safe_open(safetensor_file, framework="pt", device="cpu") as f:
            # Check each key in the weight map
            for key, file_path in weight_map.items():
                if file_path == safetensor_file:
                    if key in f.keys():
                        tensor = f.get_tensor(key)
                        dtype_bit_size = tensor.element_size() * 8
                        tensor_size = tensor.numel()
                        if tensor.dim() == 2:
                            tensor_size = tensor.shape[0] * tensor.shape[1]  # Product of rows and columns if matrix
                        # Calculate the total bytes for this tensor and store it
                        tensors_info[key] = tensor_size * dtype_bit_size / 8

    # Sum all the calculated tensor sizes
    total_memory_usage = sum(tensors_info.values())

    # Return total memory usage
    return total_memory_usage