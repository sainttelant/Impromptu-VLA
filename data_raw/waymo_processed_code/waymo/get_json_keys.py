import json

# 加载 JSON 文件
with open('/data_storage/gaoha/Downloads/waymo/validation/extracted_labels/scenario_annotations.json',  encoding='utf-8') as file:
    data = json.load(file)
   

# 假设 JSON 是一个字典，提取所有键（属性名）
def extract_keys(data):
    keys = set()
    if isinstance(data, dict):
        for key, value in data.items():
            keys.add(key)
            keys.update(extract_keys(value))
    elif isinstance(data, list):
        for item in data:
            keys.update(extract_keys(item))
    return keys

# 提取属性名
keys = extract_keys(data)

# 打印所有属性名
# 过滤掉长度大于 20 的键
filtered_keys = {key for key in keys if len(key) < 20}

# 打印过滤后的键
for key in filtered_keys:
    print(key)
