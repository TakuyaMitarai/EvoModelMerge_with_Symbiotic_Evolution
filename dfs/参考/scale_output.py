import torch
from safetensors.torch import safe_open

# テンソルを保持するディクショナリを初期化
tensors = {}

# safetensorsファイルを開いて特定のテンソルを読み込む
with safe_open("model-00001-of-00004.safetensors", framework="pt", device="cpu") as f:
    # 特定のキーに対してのみ処理
    keys_of_interest = ["model.input_layers", "model.input_scales"]
    for k in keys_of_interest:
        if k in f.keys():  # ファイルにキーが存在するか確認
            # 各キーに対応するテンソルをディクショナリに格納
            tensors[k] = f.get_tensor(k)
            # キーとテンソルのサイズを出力
            print(f"Key: {k}, Tensor size: {tensors[k].size()}, Tensor data: {tensors[k]}")
        else:
            print(f"Key {k} not found in the safetensors file.")

