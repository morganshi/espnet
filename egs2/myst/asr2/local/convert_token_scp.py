import json, sys

input_dir=sys.argv[1]
output_dir=sys.argv[2]
# 输入和输出文件名
input_file = input_dir+"/tokens.json"
output_file = output_dir+"/speech_token.scp"

# 读取 JSON 文件
with open(input_file, "r") as f:
    lines = f.readlines()


def flatten(lst):
    """ 递归展开任意层数的嵌套列表 """
    if isinstance(lst, list):
        return [x for sublist in lst for x in flatten(sublist)]
    else:
        return [lst]  # 不是列表，直接返回

data = {}
for line in lines:
    obj = json.loads(line.strip())  # 解析 JSON
    for key, value in obj.items():
        data[key] = " ".join(map(str, flatten(value)))  # 递归展开并转换为字符串

# 按 ID 排序
sorted_data = sorted(data.items())

# 写入到新文件
with open(output_file, "w") as f:
    for key, numbers in sorted_data:
        f.write(f"{key} {numbers}\n")


print(f"Processed data saved to {output_file}")
