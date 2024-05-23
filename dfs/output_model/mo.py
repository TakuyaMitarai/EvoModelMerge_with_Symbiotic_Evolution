import json
import re

# 入力ファイルパスと出力ファイルパス
input_file_path = '/Users/takuyam/Documents/workspace/EvoMerge/EvoModelMerge_with_Symbiotic_Evolution/dfs/output_model/1model.safetensors.index.json'
output_file_path = '/Users/takuyam/Documents/workspace/EvoMerge/EvoModelMerge_with_Symbiotic_Evolution/dfs/output_model/1model.safetensors.index.json'

# JSONファイルを読み込む
with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 正規表現パターン
pattern = re.compile(r'model\.layers\.(\d+)')

# weight_mapのキーを置換
new_weight_map = {}
for key, value in data['weight_map'].items():
    new_key = re.sub(pattern, lambda match: f"model.layers.{int(match.group(1)) + 64 + 32}", key)
    new_weight_map[new_key] = value

# 新しいweight_mapを設定
data['weight_map'] = new_weight_map

# 新しいJSONファイルに書き出す
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=2)
