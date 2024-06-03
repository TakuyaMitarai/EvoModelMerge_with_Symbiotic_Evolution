import os
import re

# ファイルパスのテンプレート
input_file_template_1 = '~/.cache/huggingface/hub/models--SakanaAI--EvoLLM-JP-v1-10B/snapshots/78cad5aad0897f75df8b6ee17983de0be133eb0f/0model-000{}-of-00003.safetensors'
output_file_template_1 = '~/.cache/huggingface/hub/models--SakanaAI--EvoLLM-JP-v1-10B/snapshots/78cad5aad0897f75df8b6ee17983de0be133eb0f/0model-000{}-of-00003.safetensors'
input_file_template_2 = '~/.cache/huggingface/hub/models--SakanaAI--EvoLLM-JP-v1-10B/snapshots/78cad5aad0897f75df8b6ee17983de0be133eb0f/1model-000{}-of-00004.safetensors'
output_file_template_2 = '~/.cache/huggingface/hub/models--SakanaAI--EvoLLM-JP-v1-10B/snapshots/78cad5aad0897f75df8b6ee17983de0be133eb0f/1model-000{}-of-00004.safetensors'

# 置換関数
def replace_func(match):
    number = int(match.group(1)) + 64
    return f'model.layers.{number}'.encode()

def replace_func2(match):
    number = int(match.group(1)) + 64 + 32
    return f'model.layers.{number}'.encode()

# 正規表現パターン
pattern = re.compile(rb'model\.layers\.(\d+)')

def process_files(input_file_template, output_file_template, start, end, replace_function):
    for i in range(start, end + 1):
        input_file_path = os.path.expanduser(input_file_template.format(str(i).zfill(2)))
        output_file_path = os.path.expanduser(output_file_template.format(str(i).zfill(2)))

        # ファイルをバイナリモードで開く
        with open(input_file_path, 'rb') as file:
            first_line = file.readline()
            rest_of_file = file.read()

        # 置換
        modified_first_line = re.sub(pattern, replace_function, first_line)

        # 新しいファイルに書き出し
        with open(output_file_path, 'wb') as new_file:
            new_file.write(modified_first_line)
            new_file.write(rest_of_file)

# 00001 ~ 00003までのファイルを処理
process_files(input_file_template_1, output_file_template_1, 1, 3, replace_func)

# 00001 ~ 00004までのファイルを処理
process_files(input_file_template_2, output_file_template_2, 1, 4, replace_func2)
