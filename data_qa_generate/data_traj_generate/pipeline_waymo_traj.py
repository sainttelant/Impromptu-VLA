"""
@file   参考waymo_parse_tfrecord_main.py这个代码读取数据
@brief  Waymo自车坐标系处理器 (支持轨迹差分计算)
"""

# 一个场景一个夹
import os
import sys
import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
# print(project_root)
data_dir_train = project_root / "data_raw" / "waymo"/"training"
data_dir_val = project_root / "data_raw" / "waymo"/"validation"
output_file_train =Path( project_root / "data_qa_generate" / "data_traj_generate" / "data_traj_results" / "waymo"/"traj_waymo_train.jsonl")
output_file_val= Path(project_root / "data_qa_generate" / "data_traj_generate" / "data_traj_results" / "waymo"/"traj_waymo_val.jsonl")
output_file_train.parent.mkdir(parents=True, exist_ok=True)
output_file_val.parent.mkdir(parents=True, exist_ok=True)
raw_hz=10
step_by_hz=raw_hz//2

class PipelineProcessor:
    """处理整个数据管道"""
    def __init__(self, input_paths, output_paths, max_prev=5, max_next=6, workers=4):

        if len(output_paths) != 1:
            raise ValueError("Output paths must contain exactly one file")
            
        self.input_paths = input_paths
        self.output_path = output_paths[0]
        self.max_prev = max_prev
        self.max_next = max_next
        self.workers = workers

    def _parse_frame(self, data):
        """解析帧数据"""
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        return frame

    def _format_coordinates(self, coord_list):
        """格式化坐标，保留两位小数"""
        return [round(coord, 2) for coord in coord_list]

    def _process_single_file(self, input_path):
        """处理单个文件,这个版本是生成前三个和后十个相对，x前，y左"""
        dataset = tf.data.TFRecordDataset(input_path, compression_type='')
        all_frames = [self._parse_frame(data) for data in dataset]
        
        # 采样频率
        step_by_hz = 5
        indices = [i for i in range(0, len(all_frames), step_by_hz)]
        subsscene_data = [all_frames[i] for i in indices]
        
        seq = input_path.split("/")[-1].split(".")[0]
        
        # 获取位姿信息
        raw_poses = [np.array(frame.pose.transform).reshape(4, 4).astype(np.float32) for frame in subsscene_data]
        re_poses = [np.linalg.inv(raw_pose) for raw_pose in raw_poses]
        
        # 保存该序列的所有相对位姿
        sequence_data = {
            "sequence_id": seq,
            "num_frames": len(raw_poses),
            "relative_poses": {}
        }
        
        # 对于每一帧，计算其相对于前3帧和后10帧的相对位姿
        for j in range(len(raw_poses)):
            sequence_data["relative_poses"][str(j)] = {}
            
            # 前3帧
            for i in range(max(0, j-3), j):
                rel_pose = re_poses[j] @ raw_poses[i]
                sequence_data["relative_poses"][str(j)][f"prev_{j-i}"] = self._format_coordinates(rel_pose[:2, 3].tolist())
            
            # 后10帧
            for i in range(j+1, min(len(raw_poses), j+11)):
                rel_pose = re_poses[j] @ raw_poses[i]
                sequence_data["relative_poses"][str(j)][f"next_{i-j}"] = self._format_coordinates(rel_pose[:2, 3].tolist())
        
        return sequence_data


    def execute(self):
        """执行处理流程"""
        with open(self.output_path, 'w') as f:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = [executor.submit(self._process_single_file, path) for path in self.input_paths]
                for future in tqdm(futures, desc="Processing files"):
                    json.dump(future.result(), f)
                    f.write('\n')  # 每行一个JSON对象

    def _convert_to_json_serializable(self, data):
        """数据类型转换"""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in data]
        elif isinstance(data, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in data.items()}
        return data




if __name__ == "__main__":
    configs = [
        {
            'data_split': 'validation',
            'input_dir': data_dir_val,
            'output_path': output_file_val
        },

        {
            'data_split': 'training',
            'input_dir': data_dir_train,
            'output_path': output_file_train
        }
    ]

    for config in configs:
        print(f"Processing {config['data_split']} data from {config['input_dir']}")
        input_paths = sorted(glob(os.path.join(config['input_dir'], "*.tfrecord")))
        processor = PipelineProcessor(
            input_paths=input_paths,
            output_paths=[config['output_path']],
            max_prev=0,
            max_next=9,
            workers=32
        )
        processor.execute()
        print(f"Finished processing {config['data_split']} data")