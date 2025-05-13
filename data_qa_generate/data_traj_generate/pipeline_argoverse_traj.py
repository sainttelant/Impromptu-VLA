import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
data_dir = project_root / "data_raw" / "Argoverse-V2-sensor"
modes = ['test', 'val', 'train']
     
# frame中的tx,ty,tz是ENU的
class ArgoversePlanning:
    def __init__(self,max_previous_samples,max_next_samples,output_file,mode):
        self.mode = mode
        self.base_dir = data_dir / mode
        self.output_file=output_file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.max_previous_samples=max_previous_samples
        self.max_next_samples=max_next_samples
        self.time_interval = 0.5 * 1e9  # 5ms间隔

    def process(self):
        """主处理流程"""
        all_frames = []
        csv_files = self.find_csv_files()
        
        for csv_file in csv_files:
            try:
                scene_id = self.get_scene_id(csv_file)
                df = self.read_csv_data(csv_file)
                scene_data = self.process_scene(df, scene_id)
                all_frames.extend(scene_data)
            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")
        
        self.save_results(all_frames)
        print(f"Processed {len(all_frames)} frames.")

    def find_csv_files(self):
        """查找轨迹文件"""
        return sorted(self.base_dir.rglob("city_SE3_egovehicle.csv"))

    def get_scene_id(self, csv_file):
        """获取场景ID"""
        return csv_file.parent.name

    def read_csv_data(self, csv_file):
        """读取CSV数据"""
        df = pd.read_csv(csv_file)
        df['timestamp_ns'] = pd.to_datetime(
            df['timestamp_ns'], 
            format='%Y-%m-%d %H:%M:%S.%f'
        ).astype('int64')
        return df.sort_values('timestamp_ns')

    def process_scene(self, df, scene_id):
        """处理单个场景"""
        frames = []
        timestamps = df['timestamp_ns'].values
        num_samples = self.count_images(scene_id)
        target_times = self.generate_target_times(timestamps[0], num_samples)
        selected_indices = self.align_timestamps(timestamps, target_times)
        df_subset = df.iloc[selected_indices]
        # print(len(selected_indices))
        for idx in range(len(selected_indices)): 
            if idx >= len(df_subset):
                continue
                
            frame_data = df_subset.iloc[idx]
            R = self.get_rotation_matrix(frame_data)
            trajectory=self.get_trajectory(df_subset, idx, R)
            
            frames.append({
                "scene_id": scene_id,
                "frame_id":idx,
                "pose": self.build_pose(frame_data, R),
                "trajectory":trajectory,
                # "df_trajectory": diff_traj,
                "timestamp": frame_data['timestamp_ns'] / 1e9
            })
      
        return frames

    def count_images(self, scene_id):
        """统计图片数量"""
        img_dir = self.base_dir / scene_id / "sensors" / "cameras" / "ring_front_center"
        return len(list(img_dir.glob("*")))

    def generate_target_times(self, start_time, num_samples):
        """生成目标时间序列"""
        return np.linspace(
            start_time, 
            start_time + (num_samples-1)*self.time_interval,
            num=num_samples
        )

    def align_timestamps(self, timestamps, targets):
        """时间戳对齐"""
        indices = []
        for t in targets:
            idx = np.searchsorted(timestamps, t, side="left")
            if idx == 0:
                indices.append(0)
            elif idx >= len(timestamps):
                indices.append(len(timestamps)-1)
            else:
                if (t - timestamps[idx-1]) < (timestamps[idx] - t):
                    indices.append(idx-1)
                else:
                    indices.append(idx)
        return sorted(list(set(indices)))

    def get_rotation_matrix(self, frame_data):
        """构造旋转矩阵"""
        qw, qx, qy, qz = frame_data[['qw','qx','qy','qz']]
        return np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ]).T  # 直接返回转置矩阵(frame中的四元数是自车-》ENU的)

    def get_df_trajectory(self, df, center_idx, R):
        """获取滑动窗口轨迹"""
        start = max(0, center_idx - self.max_previous_samples-1)
        end = min(len(df), center_idx + self.max_next_samples + 1)
        # 需要转换到自车坐标系（正x是向右，正y是向前）
        trajectory = [
            (R @ np.array([row.tx_m, row.ty_m, row.tz_m]))[:2] 
            for row in df.iloc[start:end].itertuples()
        ]

        """计算差分轨迹"""
        # 如果轨迹长度小于 2，直接返回零差分
        if len(trajectory) < 2:
            return np.array([0, 0]).tolist(),np.array([0, 0]).tolist()
        
        # 使用 NumPy 计算差分
        trajectory = np.array(trajectory)  # 确保输入为 NumPy 数组
        disp_trajectory = np.diff(trajectory, axis=0)
        
        # 如果 start_idx 为 0，在第一帧前补零
        if start == 0:
            disp_trajectory = np.vstack(([0, 0], disp_trajectory))

        # 交换 x 和 y，并将 y 取反
        disp_trajectory = disp_trajectory[:, [1, 0]]  # 交换 x 和 y
        disp_trajectory[:, 0] = -disp_trajectory[:, 0]  # 将新的 x（原来的 y）取反。
        
        return disp_trajectory.tolist()

    def get_trajectory(self, df, center_idx, R):
        """获取滑动窗口轨迹（新坐标系下，且相对于中心点）"""
        start = max(0, center_idx - self.max_previous_samples)
        end = min(len(df), center_idx + self.max_next_samples + 1)
        
        # 获取中心点在新坐标系下的坐标
        center_row = df.iloc[center_idx]
        vec_center = R @ np.array([center_row.tx_m, center_row.ty_m, center_row.tz_m])
        center_x, center_y = vec_center[1], vec_center[0]  # 新坐标系下的中心坐标
        
        # 生成轨迹并转换为相对于中心点的坐标
        trajectory = [
            (vec[0] - center_y, vec[1] - center_x)  
            for row in df.iloc[start:end].itertuples()
            for vec in [R @ np.array([row.tx_m, row.ty_m, row.tz_m])]
        ]
        trajectory=self.clean_trajectory(trajectory, self.max_next_samples, self.max_previous_samples)
        return trajectory

    def clean_trajectory(self,trajectory, max_next_samples, max_previous_samples):   
        trajectory = [list(point) for point in trajectory] 
        if len(trajectory) > 1:
            if max_next_samples == 0:    
                # print(trajectory[-1])    
                if trajectory[-1] == [0.0, -0.0]:
                    trajectory = trajectory[:-1]             
            if max_previous_samples == 0:        
                if trajectory[0] == [0.0, -0.0]:
                    trajectory = trajectory[1:]
        trajectory = np.array(trajectory)
        trajectory = np.round(trajectory, 2).tolist()
        return trajectory
        

    def build_pose(self, frame_data, R):
        """构建pose信息"""
        vehicle_pos = R @ np.array([frame_data.tx_m, frame_data.ty_m, frame_data.tz_m])
        return [
            frame_data.qw, frame_data.qx,
            frame_data.qy, frame_data.qz,
            vehicle_pos[0], vehicle_pos[1], vehicle_pos[2]
        ]

    def save_results(self, data):
        """保存结果"""
        with open(self.output_file, 'w') as f:
            for entry in data:
                json.dump(entry, f)
                f.write('\n')
            print("output:",self.output_file)
    

if __name__ == "__main__":
    for mode in modes:
        print(f"Processing mode: {mode}")
        output_file_last = Path(project_root / "data_qa_generate" / "data_traj_generate" / "data_traj_results" / "argoverse" / f"traj_argoverse_{mode}_last3.jsonl")
        output_file_future = Path(project_root / "data_qa_generate" / "data_traj_generate" / "data_traj_results" / "argoverse" / f"traj_argoverse_{mode}_future10.jsonl")
   
        print("Processing history samples (last 3)...")
        processor_history = ArgoversePlanning(
            max_previous_samples=3,
            max_next_samples=0,
            output_file=output_file_last,
            mode=mode
        )
        processor_history.process()
        
        # Second run: 10 future samples
        print("Processing future samples (next 10)...")
        processor_future = ArgoversePlanning(
            max_previous_samples=0,
            max_next_samples=10,
            output_file=output_file_future,
            mode=mode
        )
        processor_future.process()