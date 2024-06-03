import json


def generate_safetensors_index(idx_to_dic_input_layer, total_size):
    """Generate the 'model.safetensors.index.json' file based on vector V and total_size."""
    # Attribute list
    attributes = [
        "post_attention_layernorm.weight",
        "self_attn.k_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_proj.weight",
        "self_attn.v_proj.weight",
        "input_layernorm.weight",
        "mlp.down_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
    ]

    # Initialize JSON data structure with metadata and weight_map
    index_data = {
        "metadata": {"total_size": int(total_size)},
        "weight_map": {
            "lm_head.weight": "0model-00003-of-00003.safetensors",
            "model.embed_tokens.weight": "0model-00001-of-00003.safetensors",
            "model.input_layers": "model-layer_info.safetensors",
            "model.input_scales": "model-layer_info.safetensors",
        },
    }

    # Load 0model and 1model JSON files
    with open("output_model/0model.safetensors.index.json", "r") as file:
        data_0 = json.load(file)

    with open("output_model/1model.safetensors.index.json", "r") as file:
        data_1 = json.load(file)

    V = [i for i in range(len(idx_to_dic_input_layer))]
    # Generate entries in the weight_map for each layer and attribute
    for layer in V:
        for attr in attributes:
            key = f"model.layers.{layer}.{attr}"
            # Check if key exists in 0model and 1model, prioritize 0model
            if key in data_0["weight_map"]:
                index_data["weight_map"][key] = "0" + data_0["weight_map"][key]
            elif key in data_1["weight_map"]:
                index_data["weight_map"][key] = "1" + data_1["weight_map"][key]

    index_data["weight_map"]["model.norm.weight"] = "0model-00003-of-00003.safetensors"

    # Save as JSON file
    with open('.cache/huggingface/hub/models--SakanaAI--EvoLLM-JP-v1-10B/snapshots/78cad5aad0897f75df8b6ee17983de0be133eb0f/model.safetensors.index.json', 'w') as json_file:
        json.dump(index_data, json_file, indent=2)

    # Return the generated data for verification or testing purposes
    return index_data
