import json
import random

def generate_safetensors_index(V, total_size):
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
        "mlp.up_proj.weight"
    ]

    # Initialize JSON data structure with metadata and weight_map
    index_data = {
        "metadata": {
            "total_size": total_size
        },
        "weight_map": {
            "lm_head.weight": "model-00004-of-00004.safetensors",
            "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
            "model.input_layers": "model-layer_info.safetensors",
            "model.input_scales": "model-layer_info.safetensors",
        }
    }

    # Generate entries in the weight_map for each layer and attribute
    for layer in V:
        for attr in attributes:
            key = f"model.layers.{layer}.{attr}"
            file_index = layer + 1
            file_name = f"model-{file_index:05d}-of-00004.safetensors"
            index_data["weight_map"][key] = file_name

    # Add fixed entry
    index_data["weight_map"]["model.norm.weight"] = "model-00004-of-00004.safetensors"

    # Save as JSON file
    with open('model.safetensors.index.json', 'w') as json_file:
        json.dump(index_data, json_file, indent=2)

    # Return the generated data for verification or testing purposes
    return index_data
