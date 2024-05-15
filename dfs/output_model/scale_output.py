import torch
from safetensors.torch import save_file, safe_open

def output_layer_info(input_layers, input_scales):
    data_to_save = {"input_layers": input_layers, "input_scales": input_scales}
    save_path = "/root/.cache/huggingface/hub/models--new_model/   /model-layer_info.safetensors"
    save_file(data_to_save, save_path)

    # # テンソルを保持するディクショナリを初期化
    # tensors = {}

    # # safetensorsファイルを開いてテンソルを読み込む
    # with safe_open("model-scale.safetensors", framework="pt", device="cpu") as f:
    #     # ファイル内の全てのキーに対してループ
    #     for k in f.keys():
    #         # 各キーに対応するテンソルをディクショナリに格納
    #         tensors[k] = f.get_tensor(k)
    #         # キーとテンソルのサイズを出力
    #         print(f"Key: {k}, Tensor size: {tensors[k].size()}, Tensor data: {tensors[k]}")
