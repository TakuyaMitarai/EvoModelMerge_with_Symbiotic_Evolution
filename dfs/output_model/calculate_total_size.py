import json
import torch
from safetensors.torch import safe_open
import os

def total_size():
    base_path = os.path.expanduser( "~/.cache/huggingface/hub/models--SakanaAI--EvoLLM-JP-v1-10B/snapshots/78cad5aad0897f75df8b6ee17983de0be133eb0f/")
    index_file_path = base_path + "model.safetensors.index.json"

    # Indexファイルをロードしてテンソル名とsafetensorsファイルのマッピングを取得
    with open(index_file_path, "r") as file:
        index_data = json.load(file)
        weight_map = index_data["weight_map"]

    # 開く必要があるファイルパスの辞書
    files_to_open = {base_path + file for file in weight_map.values()}

    # テンソルデータを収集するための辞書
    tensors_info = {}

    # 各safetensorsファイルを開き、必要なテンソルデータを読み込む
    for safetensor_file in files_to_open:
        print(safetensor_file)
        with safe_open(safetensor_file, framework="pt", device="cpu") as f:
            # weight_map内の各キーをチェック
            for key, file_path in weight_map.items():
                full_file_path = base_path + file_path
                if full_file_path == safetensor_file:
                    if key in f.keys():
                        tensor = f.get_tensor(key)
                        dtype_bit_size = tensor.element_size() * 8
                        tensor_size = tensor.numel()
                        if tensor.dim() == 2:
                            tensor_size = tensor.shape[0] * tensor.shape[1]  # 行と列の積（行列の場合）
                        # このテンソルの総バイト数を計算して格納
                        tensors_info[key] = tensor_size * dtype_bit_size / 8

    # 計算されたすべてのテンソルサイズを合計
    total_memory_usage = sum(tensors_info.values())

    # 総メモリ使用量を返す
    return total_memory_usage
