import re

# ファイルパス
input_file_path = 'model-00001-of-00004.safetensors'
output_file_path = 'model-00001-of-00004-modified.safetensors'

# ファイルをバイナリモードで開く
with open(input_file_path, 'rb') as file:
    # 最初の行だけ読み込む
    first_line = file.readline()
    # 残りの内容を保存
    rest_of_file = file.read()

# 文字列置換を行う正規表現
pattern = re.compile(rb'model\.layers\.(\d+)')

# 必要な置換を行う関数
def replace_func(match):
    # キャプチャされた数字に32を加算
    number = int(match.group(1)) + 32
    # 新しい文字列を返す
    return f'model.layers.{number}'.encode()

# 最初の行に対して置換を行う
modified_first_line = re.sub(pattern, replace_func, first_line)

# 新しいファイルに書き出す
with open(output_file_path, 'wb') as new_file:
    # 置換された最初の行を書き込む
    new_file.write(modified_first_line)
    # 残りのファイル内容を書き込む
    new_file.write(rest_of_file)

