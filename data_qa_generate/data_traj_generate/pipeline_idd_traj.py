import os
import json
import pandas as pd
import numpy as np
from pyproj import Transformer
from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
# print(project_root)
data_dir = project_root / "data_raw" / "idd_multimodal" / "primary"
output_file_last =Path( project_root / "data_qa_generate" / "data_traj_generate" / "data_traj_results" / "idd"/"traj_idd_last3.jsonl")
output_file_future = Path(project_root / "data_qa_generate" / "data_traj_generate" / "data_traj_results" / "idd"/"traj_idd_future10.jsonl")
output_file_last.parent.mkdir(parents=True, exist_ok=True)
output_file_future.parent.mkdir(parents=True, exist_ok=True)

def convert_latlon_to_utm(lons, lats, epsg_target="EPSG:32644"):
    """
    将经纬度数组转换为UTM坐标。
    :param lons: 经度数组
    :param lats: 纬度数组
    :param epsg_target: 目标坐标系，此处默认为UTM Zone 44N
    :return: (eastings, northings)
    """
    transformer = Transformer.from_crs("EPSG:4326", epsg_target, always_xy=True)
    eastings, northings = transformer.transform(lons, lats)
    return eastings, northings

class IDDTrajectoryProcessor:
    def __init__(self, max_previous_samples=0, max_next_samples=10, output_file=None):
        self.data_dir = data_dir
        self.sub_dirs = ['d0', 'd1', 'd2']
        self.dataset_types = ['train', 'val']
        self.output_file = output_file
        self.max_previous_samples = max_previous_samples
        self.max_next_samples = max_next_samples
        self.results = []
        self.rotation_cache = None

    def process(self):
        for sub_dir in self.sub_dirs:
            for dataset_type in self.dataset_types:
                file_path = self.data_dir / sub_dir / f"{dataset_type}.csv"
                if not file_path.exists():
                    print(f"文件 {file_path} 不存在，跳过。")
                    continue
                print(f"正在处理文件：{file_path}")
                df = pd.read_csv(file_path)

                # 利用经纬度数据解析得到UTM坐标（存储为 easting 和 northing 列）
                lons = df['longitude'].values
                lats = df['latitude'].values
                eastings, northings = convert_latlon_to_utm(lons, lats)
                df['easting'] = eastings
                df['northing'] = northings

                # 计算相对于文件第一个点的相对坐标
                easting_ref, northing_ref = eastings[0], northings[0]
                df['x_rel'] = df['easting'] - easting_ref
                df['y_rel'] = df['northing'] - northing_ref

                # 生成符合2Hz的索引序列（交替+8/+9）
                indices = []
                current = 0
                add_eight = True
                while current < len(df):
                    indices.append(current)
                    step = 7 if add_eight else 8
                    next_current = current + step
                    if next_current >= len(df):
                        break
                    current = next_current
                    add_eight = not add_eight

                # 创建降采样子集
                df_subset = df.loc[indices].reset_index(drop=True)  # 重置索引保证连续
                traj = df_subset[['easting', 'northing']].values
                self.compute_traj_to_thetas(traj)  # 基于子集预计算航向角

                # 处理每个选中的帧
                for i in range(len(df_subset)):
                    original_idx = indices[i]  # 获取原始df中的索引
                    try:
                        # 使用子集数据和连续索引计算轨迹
                        traj, diff_traj = self.get_trajectory(df_subset, i)

                        frame_data = {
                            "scene_id": f"{sub_dir}_{dataset_type}",
                            "frame_id": original_idx,  # 保留原始索引作为frame_id
                            "image_idx": str(df.loc[original_idx, "image_idx"]),
                            "timestamp": str(df.loc[original_idx, "timestamp"]),
                            "trajectory": traj
                        }
                        self.results.append(frame_data)
                    except Exception as e:
                        print(f"处理 {sub_dir}_{dataset_type} 中帧 {original_idx} 时出错：{str(e)}")
                        continue

        self.save_results(self.results)
        print(f"处理完成，总帧数：{len(self.results)}")

    def compute_traj_to_thetas(self, traj):
        """
        根据UTM轨迹数据计算每个点对应的航向角（单位：角度）。
        若相邻点距离小于0.2，则角度置为0。
        :param traj: numpy 数组，形状 (N, 2)（[easting, northing]）
        :return: numpy 数组 [N]，每个点对应的航向角（角度制）
        """
        traj = np.array(traj)
        N = len(traj)
        if N < 2:
            thetas = np.zeros(N)
            self.rotation_cache = thetas
            return thetas
        diffs = np.diff(traj, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        angles = np.degrees(np.arctan2(diffs[:, 0], diffs[:, 1]))  # 参数顺序改为 (x, y),这样的话，是ENU，北0度，东90度
        angles[dists < 0.2] = 0
        thetas = np.empty(N)
        thetas[0] = angles[0]
        thetas[1:] = angles
        self.rotation_cache = thetas
        return thetas

    def get_trajectory(self, df, center_idx):
        """
        获取滑动窗口内的轨迹，并转换到当前帧的车辆坐标系下（以当前帧的UTM坐标为原点，使用该帧航向角旋转）。
        :param df: 包含 'easting' 和 'northing' 列的 DataFrame
        :param center_idx: 当前帧索引
        :return:
            trajectory: 滑动窗口内转换后的轨迹点列表
            diff_traj: 相邻轨迹点之间的差分轨迹
        """
        # 当前帧的航向角（单位：角度），转换为弧度
        theta = self.rotation_cache[center_idx]
        heading_radians = np.deg2rad(theta)
        # 构造旋转矩阵，将全局UTM坐标转换到车辆坐标系（车辆正前方为 y 轴正方向）
        R = np.array([
            [np.cos(heading_radians), -np.sin(heading_radians)],
            [np.sin(heading_radians), np.cos(heading_radians)]
        ])
        # 以当前帧的UTM坐标作为原点
        origin = df[['easting', 'northing']].iloc[center_idx].values

        # 确定滑动窗口范围
        start = max(0, center_idx - self.max_previous_samples)
        end = min(len(df), center_idx + self.max_next_samples+1)
        points = df[['easting', 'northing']].iloc[start:end].values
        # 转换：计算相对于当前帧原点的偏移，再应用旋转
        transformed = (R @ (points - origin).T).T
        trajectory = transformed.tolist()
        
        # # 计算差分轨迹：相邻点差分
        traj_array = np.array(trajectory)
        if len(traj_array) < 2:
            diff_traj = [[0.0, 0.0]]
        else:
            diff = np.diff(traj_array, axis=0)
            # 如果滑动窗口首点为当前帧，则在差分前补 [0,0]
            if start == 0:
                diff_traj = np.vstack(([0.0, 0.0], diff)).tolist()
            else:
                diff_traj = diff.tolist()
        trajectory=self.clean_trajectory(transformed, self.max_next_samples, self.max_previous_samples)
        return trajectory, diff_traj

    def clean_trajectory(self,transformed, max_next_samples, max_previous_samples):   
        transformed[:, 0], transformed[:, 1] = transformed[:, 1], -transformed[:, 0]
        trajectory = transformed.tolist()      
        if len(trajectory) > 1:
            if max_next_samples == 0:        
                if trajectory[-1] == [0.0, -0.0]:
                    trajectory = trajectory[:-1]             
            if max_previous_samples == 0:        
                if trajectory[0] == [0.0, -0.0]:
                    trajectory = trajectory[1:]
        trajectory = np.array(trajectory)
        trajectory = np.round(trajectory, 2).tolist()
        return trajectory

    def save_results(self, data):
        with open(self.output_file, 'w') as f:
            for entry in data:
                json.dump(entry, f)
                f.write('\n')
            print("output:",self.output_file)


if __name__ == "__main__":
    # First run: 3 history samples
    processor_history = IDDTrajectoryProcessor(
        max_previous_samples=3,
        max_next_samples=0,
        output_file=output_file_last
    )
    processor_history.process()
    
    # Second run: 10 future samples
    processor_future = IDDTrajectoryProcessor(
        max_previous_samples=0,
        max_next_samples=10,
        output_file=output_file_future
    )
    processor_future.process()