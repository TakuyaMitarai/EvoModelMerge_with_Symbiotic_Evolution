import torch
from safetensors.torch import save_file, safe_open
import os

def output_layer_info(input_layers, input_scales):
    # 入力データをテンソルに変換
    input_layers_tensor = torch.tensor(input_layers)
    input_scales_tensor = torch.tensor(input_scales)

    data_to_save = {"input_layers": input_layers_tensor, "input_scales": input_scales_tensor}
    save_path = os.path.expanduser("~/.cache/huggingface/hub/models--SakanaAI--EvoLLM-JP-v1-10B/snapshots/78cad5aad0897f75df8b6ee17983de0be133eb0f/model-layer_info.safetensors")
    save_file(data_to_save, save_path)

    # テンソルを保持するディクショナリを初期化
    tensors = {}

    # safetensorsファイルを開いてテンソルを読み込む
    with safe_open(save_path, framework="pt", device="cpu") as f:
        # ファイル内の全てのキーに対してループ
        for k in f.keys():
            # 各キーに対応するテンソルをディクショナリに格納
            tensors[k] = f.get_tensor(k)
            # キーとテンソルのサイズを出力
            print(f"Key: {k}, Tensor size: {tensors[k].size()}, Tensor data: {tensors[k]}")