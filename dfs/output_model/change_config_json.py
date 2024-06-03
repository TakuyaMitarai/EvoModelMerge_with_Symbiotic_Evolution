import json


def update_config(num_hidden_layers: int, num_hops: int):
    file_path = "~/.cache/huggingface/hub/models--SakanaAI--EvoLLM-JP-v1-10B/snapshots/78cad5aad0897f75df8b6ee17983de0be133eb0f/config.json"  # ファイルパスをここで指定

    # JSONファイルの内容を読み込む
    with open(file_path, "r") as file:
        config = json.load(file)

    # num_hidden_layersとnum_hopsの値を引数の値に置き換える
    config["num_hidden_layers"] = num_hidden_layers
    config["num_hops"] = num_hops

    # 更新した内容をJSONファイルに書き込む
    with open(file_path, "w") as file:
        json.dump(config, file, indent=2)
