import json
import re

def process_files(idx_to_dic_input_layer):
    # Load 0model and 1model JSON files
    with open("output_model/0model.safetensors.index.json", "r") as file:
        data_0 = json.load(file)

    with open("output_model/1model.safetensors.index.json", "r") as file:
        data_1 = json.load(file)

    # Merge the weight maps, prioritizing data_0
    weight_map = data_0["weight_map"].copy()
    weight_map.update({k: v for k, v in data_1["weight_map"].items() if k not in weight_map})

    for i in range(len(idx_to_dic_input_layer)):
        layer_idx = idx_to_dic_input_layer[i] + 64
        key = f"model.layers.{layer_idx}"

        if key in weight_map:
            path = weight_map[key]
            file_path = f"/root/.cache/huggingface/hub/models--SakanaAI--EvoLLM-JP-v1-10B/snapshots/78cad5aad0897f75df8b6ee17983de0be133eb0f/{path}"

            with open(file_path, 'rb') as file:
                first_line = file.readline()
                rest_of_file = file.read()

            # Replace model.layer.<layer_idx> with model.layer.<i>
            pattern = re.compile(rb'model\.layers\.' + str(layer_idx).encode())
            replacement = f'model.layers.{i}'.encode()
            modified_first_line = re.sub(pattern, replacement, first_line)

            # Save the modified content back to the same file
            with open(file_path, 'wb') as file:
                file.write(modified_first_line)
                file.write(rest_of_file)
