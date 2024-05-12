# サンプルのsetを作成
v = [0, 1, 2, 5, 1, 4, 6]
my_set = set(v)
dic = {}
v2 = []

# enumerateを使用してインデックスと要素を取り出す
for index, element in enumerate(my_set):
    dic[element] = index
    print(f"Index: {index}, Element: {element}")

for i in v:
    v2.append(dic[i])

print(v2)