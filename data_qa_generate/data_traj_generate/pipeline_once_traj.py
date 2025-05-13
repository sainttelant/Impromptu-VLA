import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as R
scene_sum_num=581
from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
data_dir = project_root / "data_raw"
modes = ['test', 'val', 'train']

class OnceProcessor:
    def __init__(self, base_path, output_file, max_previous_samples=0, max_next_samples=10):
        self.base_path = base_path
        self.output_file = output_file
        self.max_previous_samples = max_previous_samples
        self.max_next_samples = max_next_samples

    def process(self):
        data = self.read_input_data()
        grouped_data = self.group_by_scene(data)
        processed_data = self.process_groups(grouped_data)
        self.save_output(processed_data)

    def read_input_data(self):
        data = []
        for sequence_num in range(scene_sum_num):
            folder_name = f"{sequence_num:06d}"
            file_path = os.path.join(self.base_path, folder_name, f"{folder_name}.json")
            try:
                with open(file_path, 'r') as f:
                    file_data = json.load(f)
                    file_data['scene_id'] = folder_name  # 添加 scene_id
                    data.append(file_data)
            except FileNotFoundError:
                print(f"File {file_path} not found. Skipping.")
                continue
            except json.JSONDecodeError as e:
                print(f"JSON decode error in {file_path}: {e}")
                continue
        return data

    def group_by_scene(self, data):
        # 返回包含 scene_id 和 frames 的字典列表
        return [{'scene_id': entry['scene_id'], 'frames': entry.get('frames', [])} for entry in data]

    def process_groups(self, grouped_data):
        all_processed_scenes = []
        for group in grouped_data:
            scene_traj = self.extract_scene_trajs(group['frames'], group['scene_id'])
            all_processed_scenes.append(scene_traj)
        return all_processed_scenes

    def extract_scene_trajs(self, frames, scene_id):
        scene_traj = []
        for i, frame in enumerate(frames):
            traj = self.get_trajectory(frames, i,scene_id)
            scene_traj.append({
                "scene_id": scene_id,  # 添加 scene_id
                "frame_id": frame["frame_id"],
                "trajectory": traj,
            })
        return scene_traj

# frames 里的 "pose" 数据是 相对于场景第一帧（初始帧）的激光雷达坐标系 进行定义的。
# 激光雷达坐标系位于激光雷达传感器中心，x 轴正方向向左，y 轴正方向向后，z 轴正方向向上。
# 所以先旋转到自车坐标系，再xy都取相反数就行（换坐标轴）

    def get_df_trajectory(self, frames, index,scene_id):
        """
        提取当前帧的前后轨迹，并转换到自车坐标系
        :param frames: 包含所有帧数据的列表，每帧包含 'pose' 信息
        :param index: 当前帧的索引
        :return: 差分轨迹 (disp_trajectory)，形状为 (n, 2)
        """
        # 1. 确定轨迹的起始和结束索引
        start_idx = max(0, index - self.max_previous_samples-1)
        end_idx = min(len(frames), index + self.max_next_samples + 1)

        # 2. 获取当前帧的参考位姿
        ref_pose = frames[index]['pose']
        ref_translation = np.array(ref_pose[4:6])  # 提取 x, y

        # 检查四元数是否有效
        quat = ref_pose[:4]
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-6:  # 如果模接近零，跳过该帧或使用默认值.的确有一些pose中的数据全为0000000，全静止
            # print(f"Warning: Invalid quaternion (norm={quat_norm}) at frame {index}.scene{scene_id}. Using raw trajectory.")
            use_rotation = False
        else:
            use_rotation = True
            ref_rotation = R.from_quat(quat).as_matrix()[:2, :2]  # 提取旋转矩阵的前两行

        # 3. 提取轨迹并转换到自车坐标系（如果需要）
        trajectory = []
        for i in range(start_idx, end_idx):
            pose = frames[i]['pose']
            trans = np.array(pose[4:6])  # 提取 x, y
            if use_rotation:
                local_pos = ref_rotation.T @ (trans - ref_translation)  # 转换到自车坐标系
                local_pos = -local_pos  # xy坐标轴恰好都相反
            else:
                local_pos = trans  # 不使用旋转，直接使用原始位置
            trajectory.append(local_pos.tolist())

        # 4. 计算差分轨迹
        if len(trajectory) >= 2:
            translations = np.array(trajectory)
            disp_trajectory = np.diff(translations, axis=0)  # 计算差分
            if start_idx == 0:
                disp_trajectory = np.vstack(([0, 0], disp_trajectory))  # 第一帧差分补零
        else:
            disp_trajectory = np.array([[0, 0]])  # 单帧时返回零差分

        return disp_trajectory.tolist()

    def get_trajectory(self, frames, index,scene_id):
        """
        提取当前帧的前后轨迹，并转换到自车坐标系
        :param frames: 包含所有帧数据的列表，每帧包含 'pose' 信息
        :param index: 当前帧的索引
        :return: 轨迹 ,形状为 (n, 2)
        """
        # 1. 确定轨迹的起始和结束索引
        start_idx = max(0, index - self.max_previous_samples)
        end_idx = min(len(frames), index + self.max_next_samples + 1)

        # 2. 获取当前帧的参考位姿
        ref_pose = frames[index]['pose']
        ref_translation = np.array(ref_pose[4:6])  # 提取 x, y

        # 检查四元数是否有效
        quat = ref_pose[:4]
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-6:  # 如果模接近零，跳过该帧或使用默认值.的确有一些pose中的数据全为0000000，全静止
            # print(f"Warning: Invalid quaternion (norm={quat_norm}) at frame {index}.scene{scene_id}. Using raw trajectory.")
            use_rotation = False
        else:
            use_rotation = True
            ref_rotation = R.from_quat(quat).as_matrix()[:2, :2]  # 提取旋转矩阵的前两行

        # 3. 提取轨迹并转换到自车坐标系（如果需要）
        trajectory = []
        for i in range(start_idx, end_idx):
            pose = frames[i]['pose']
            trans = np.array(pose[4:6])  # 提取 x, y
            if use_rotation:
                local_pos = ref_rotation.T @ (trans - ref_translation)  # 转换到自车坐标系
                local_pos = -local_pos  # xy坐标轴恰好都相反
            else:
                local_pos = trans  # 不使用旋转，直接使用原始位置
            # trajectory.append(local_pos.tolist())
            # 将坐标从 (x, y) 转换为 (y, -x)
            new_x = local_pos[1]   # 原y作为新x
            new_y = -local_pos[0]  # 原x取反作为新y
            trajectory.append([new_x, new_y])  # 直接构造新坐标
        trajectory=self.clean_trajectory(trajectory, self.max_next_samples, self.max_previous_samples)
        return trajectory

    def clean_trajectory(self,trajectory, max_next_samples, max_previous_samples):   
        trajectory = [list(point) for point in trajectory] 
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
       

    def compute_diff_trajectory(self, traj):
        """计算轨迹的差分（相邻点的差分）"""
        return [[traj[i+1][0] - traj[i][0], traj[i+1][1] - traj[i][1]] for i in range(len(traj) - 1)]

    def save_output(self, processed_data):
        output_dir = os.path.dirname(self.output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for entry in processed_data:
                    for item in entry:
                        # 将每个条目转换为 JSON 字符串并写入文件
                        json_line = json.dumps(item, ensure_ascii=False)
                        f.write(json_line + '\n')  # 每行一个 JSON 对象
            print(f"Processed data saved to {self.output_file}")
        except Exception as e:
            print(f"Error saving output file: {e}")



# 使用示例
if __name__ == "__main__":
    base_path = f"{data_dir}/ONCE/3D_infos/untar/data/"
    processor = OnceProcessor(
        base_path, 
        output_file=f"{project_root}/data_qa_generate/data_traj_generate/data_traj_results/once/traj_once_future10.jsonl",
        max_previous_samples=0,max_next_samples=10)
    processor.process()

    processor = OnceProcessor(
        base_path, 
        output_file= f"{project_root}/data_qa_generate/data_traj_generate/data_traj_results/once/traj_once_last3.jsonl",
        max_previous_samples=3,max_next_samples=0)
    processor.process()
