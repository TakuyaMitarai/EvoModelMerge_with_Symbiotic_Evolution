import random
import json

# サイズ44で、0から100のランダムな整数を含むベクトルVを生成
# V = [random.randint(0, 100) for _ in range(44)]
V = [i for i in range(44)]

# 属性リストの準備
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

# metadataとweight_mapを含むJSONデータ構造を初期化
index_data = {
    "metadata": {
        "total_size": 19718152712
    },
    "weight_map": {}
}

# 固定されたエントリを追加
index_data["weight_map"].update({
    "lm_head.weight": "model-00004-of-00004.safetensors",
    "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
    "model.input_layers": "model-00001-of-00004.safetensors",
    "model.input_scales": "model-00001-of-00004.safetensors",
})

# weight_mapに対応するエントリを生成
for v in V:
    layer = v
    for attr in attributes:
        key = f"model.layers.{layer}.{attr}"
        file_index = v + 1
        file_name = f"model-{file_index:05d}-of-00004.safetensors"
        index_data["weight_map"][key] = file_name

# 最後の行を追加
index_data["weight_map"]["model.norm.weight"] = "model-00004-of-00004.safetensors"

# JSONファイルとして保存
with open('model.safetensors.index.json', 'w') as json_file:
    json.dump(index_data, json_file, indent=2)

# 結果を表示（確認用）
print(json.dumps(index_data, indent=2))
