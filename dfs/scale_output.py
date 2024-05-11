import torch
from safetensors.torch import save_file, safe_open

# モデルのパラメータサイズ
num_hops = 10

# input_layers を整数の乱数で初期化
input_layers = torch.randint(low=0, high=10, size=(num_hops,))

# input_scales を浮動小数点の乱数で初期化
input_scales = torch.rand(num_hops)

# 保存するパラメータを辞書形式で準備
data_to_save = {"input_layers": input_layers, "input_scales": input_scales}

# ファイルに書き出し
save_path = "model-scale.safetensors"
save_file(data_to_save, save_path)

# テンソルを保持するディクショナリを初期化
tensors = {}

# safetensorsファイルを開いてテンソルを読み込む
with safe_open("model-scale.safetensors", framework="pt", device="cpu") as f:
    # ファイル内の全てのキーに対してループ
    for k in f.keys():
        # 各キーに対応するテンソルをディクショナリに格納
        tensors[k] = f.get_tensor(k)
        # キーとテンソルのサイズを出力
        print(f"Key: {k}, Tensor size: {tensors[k].size()}, Tensor data: {tensors[k]}")
