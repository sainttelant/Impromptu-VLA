import os
import pickle
import json
import numpy as np

def ndarray_to_list(obj):
    """递归将 NumPy 数组转换为列表"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: ndarray_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [ndarray_to_list(item) for item in obj]
    else:
        return obj

def process_pt_file(pt_file_path, output_jsonl_path):
    """处理单个.pt文件，将其内容转换为.jsonl文件"""
    try:
        with open(pt_file_path, "rb") as f:
            data = pickle.load(f)  # 加载.pt文件内容

        # 转换数据中的 NumPy 数组为列表
        data = ndarray_to_list(data)

        # 检查数据结构
        if isinstance(data, dict):  # 如果是字典，直接写入一行
            with open(output_jsonl_path, "w") as f:
                json.dump(data, f)
                f.write("\n")
        elif isinstance(data, list):  # 如果是列表，逐行写入
            with open(output_jsonl_path, "w") as f:
                for item in data:
                    json.dump(item, f)
                    f.write("\n")
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        print(f"Processed {pt_file_path} -> {output_jsonl_path}")
    except Exception as e:
        print(f"Error processing {pt_file_path}: {str(e)}")

def convert_pt_to_jsonl(input_folder):
    """遍历文件夹，将所有.pt文件转换为.jsonl文件"""
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".pt"):
                pt_file_path = os.path.join(root, file)
                jsonl_file_name = file.replace(".pt", ".jsonl")
                output_jsonl_path = os.path.join(root, jsonl_file_name)
                process_pt_file(pt_file_path, output_jsonl_path)

# 定义输入文件夹
input_folder = "waymo_processed"

# 开始转换
convert_pt_to_jsonl(input_folder)
print("Conversion complete.")