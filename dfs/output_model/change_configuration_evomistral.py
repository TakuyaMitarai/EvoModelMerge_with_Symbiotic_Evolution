import re


def update_num_hops(nh: int):
    file_path = "/root/.cache/huggingface/hub/models--SakanaAI--EvoLLM-v1-JP-10B/snapshots/78cad5aad0897f75df8b6ee17983de0be133eb0f/configuration_evomistral.py"

    # ファイルの内容を読み込む
    with open(file_path, "r") as file:
        content = file.read()

    # num_hopsのデフォルト値を置き換える
    updated_content = re.sub(r"num_hops: int = \d+", f"num_hops: int = {nh-1}", content)

    # 更新した内容に最後の行を追加する
    if not updated_content.endswith("\n"):
        updated_content += "\n"

    # 更新した内容をファイルに書き込む
    with open(file_path, "w") as file:
        file.write(updated_content)

