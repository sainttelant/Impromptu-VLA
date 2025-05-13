"""Provides helper methods for loading and parsing KITTI data."""
"""Thanks the code:https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py"""
from collections import namedtuple  # 用于创建具有命名字段的元组
import numpy as np
from PIL import Image  # 确保安装了Pillow库

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"

# 定义OXTS数据包的命名元组，对应数据格式中的各个字段
OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +         # 纬度，经度，海拔
                        'roll, pitch, yaw, ' +       # 滚转角，俯仰角，偏航角（弧度）
                        'vn, ve, vf, vl, vu, ' +      # 北向，东向，前向，左向，上向速度（m/s）
                        'ax, ay, az, af, al, au, ' +  # 三轴加速度（m/s²）
                        'wx, wy, wz, wf, wl, wu, ' +  # 三轴角速度（rad/s）
                        'pos_accuracy, vel_accuracy, ' +  # 位置和速度精度估计
                        'navstat, numsats, ' +       # 导航状态，卫星数量
                        'posmode, velmode, orimode') # 定位模式，速度模式，方向模式

# 包含原始数据包和对应的变换矩阵（从IMU坐标系到世界坐标系）
OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')

def subselect_files(files, indices):
    """根据索引列表筛选文件列表"""
    try:
        files = [files[i] for i in indices]
    except Exception:
        pass
    return files

def rotx(t):
    """生成绕X轴的旋转矩阵"""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

def roty(t):
    """生成绕Y轴的旋转矩阵"""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def rotz(t):
    """生成绕Z轴的旋转矩阵"""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def transform_from_rot_trans(R, t):
    """从旋转矩阵和平移向量构建4x4变换矩阵"""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

def read_calib_file(filepath):
    """读取标定文件并解析为字典"""
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                key, value = line.split(':', 1)
            except ValueError:
                key, value = line.split(' ', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def pose_from_oxts_packet(packet, scale):
    """根据OXTS数据包计算SE(3)位姿矩阵"""
    er = 6378137.  # 地球赤道半径（米）

    # 使用墨卡托投影计算平移向量
    tx = scale * packet.lon * np.pi * er / 180.
    ty = scale * er * np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    # 通过欧拉角构建旋转矩阵（旋转顺序：Z-Y-X）
    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))
    return R, t

def load_oxts_packets_and_poses(oxts_files):
    """加载并解析所有OXTS数据，生成位姿信息"""
    scale = None
    origin = None
    oxts = []

    for filename in oxts_files:
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split()
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]
                packet = OxtsPacket(*line)
                if scale is None:
                    scale = np.cos(packet.lat * np.pi / 180.)
                R, t = pose_from_oxts_packet(packet, scale)
                if origin is None:
                    origin = t
                T_w_imu = transform_from_rot_trans(R, t - origin)
                oxts.append(OxtsData(packet, T_w_imu))
    return oxts

def load_image(file, mode):
    """加载图像文件并转换为指定模式（如RGB/L）"""
    return Image.open(file).convert(mode)

def yield_images(imfiles, mode):
    """生成器：逐个加载图像文件"""
    for file in imfiles:
        yield load_image(file, mode)

def load_velo_scan(file):
    """加载Velodyne点云数据（二进制格式）"""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))

def yield_velo_scans(velo_files):
    """生成器：逐个加载点云数据"""
    for file in velo_files:
        yield load_velo_scan(file)



import numpy as np
import json

class PromptKittiPlanning:
    def __init__(self):
        # 初始化配置参数
        self.input_files = [
            '/baai-cwm-1/baai_cwm_ml/algorithm/huanang.gao/shared/jianing.liu/data_tracking_oxts/training/oxts/0000.txt',
            '/baai-cwm-1/baai_cwm_ml/algorithm/huanang.gao/shared/jianing.liu/data_tracking_oxts/training/oxts/0001.txt',
            # 可根据需要添加更多文件
        ]
        self.output_path = 'output_trajectories.jsonl'
        self.max_previous_samples = 5  # 当前样本前面最多选取的样本数
        self.max_next_samples = 6      # 当前样本后面最多选取的样本数

    def process(self):
        """主处理流程"""
        data = self.read_input_data()
        grouped_data = self.group_by_scene(data)
        processed = self.process_groups(grouped_data)
        self.save_output_traj(processed)

    def read_input_data(self):
        """读取输入数据"""
        oxts_data = load_oxts_packets_and_poses(self.input_files)
        print(f"读取到 {len(oxts_data)} 条数据")
        return oxts_data

    def group_by_scene(self, data):
        """将所有数据归为一个场景"""
        return [data]  # 返回一个场景，包含所有样本

    def process_groups(self, grouped_data):
        """处理分组数据，针对每个样本计算转换到自车坐标系下的轨迹"""
        all_entries = []
        for scene_idx, scene_data in enumerate(grouped_data):
            scene_entries = self.extract_scene_trajs(scene_data)
            all_entries.extend(scene_entries)
        return all_entries

    def extract_scene_trajs(self, scene_data):
        """
        对于场景中的每个样本：
        1. 提取场景中所有样本的ENU坐标（均相对于场景第一个点）。
        2. 对于当前样本，动态选取其前面最多N个、当前样本本身和后面最多M个的数据点。
        3. 对于这些选定的点，计算相对于当前样本的位移差，并使用当前样本的yaw角
           构造旋转矩阵，将这些位移转换到自车坐标系下。
        """
        outputs = []
        # 提取场景中所有样本的ENU位置（均相对于原点）
        enu_positions = [data.T_w_imu[:3, 3] for data in scene_data]
        N = len(enu_positions)

        for i, data in enumerate(scene_data):
            current_pos = data.T_w_imu[:3, 3]
            # 从当前样本的旋转矩阵中提取yaw角（绕Z轴旋转）
            R_sample = data.T_w_imu[:3, :3]
            yaw = np.arctan2(R_sample[1, 0], R_sample[0, 0])
            # 限制 yaw 到 [-π, π] 范围
            yaw = self.normalize_angle_neg_pi_to_pi(yaw)
            # 构造将ENU坐标转换到自车坐标系的旋转矩阵（仅考虑yaw）
            R_ego = rotz(np.pi / 2 - yaw)

            # 动态选取相对于当前样本：前面最多N个、当前样本和后面最多M个
            lower_bound = max(0, i - self.max_previous_samples)
            upper_bound = min(N, i + self.max_next_samples + 1)  # 包含当前样本
            traj_indices = list(range(lower_bound, upper_bound))

            traj_sample = []
            for idx in traj_indices:
                point_pos = R_ego.dot(enu_positions[idx])
                if idx == 0:
                    disp = np.zeros_like(point_pos)
                else:
                    disp = point_pos - R_ego.dot(enu_positions[idx - 1])  # 在ENU下的差分向量
                traj_sample.append(disp[:2].tolist())  # 只取x,y两个分量

            # 获取当前样本的旋转角（roll、pitch直接从packet中获取，yaw通过矩阵计算得到）
            rotation_angles = {
                'roll': data.packet.roll,
                'pitch': data.packet.pitch,
                'yaw': yaw
            }

            outputs.append({
                'sample_index': i,
                'rotation_angles': rotation_angles,
                'trajectory': traj_sample
            })

        return outputs

    def save_output_traj(self, processed):
        """保存每个样本的旋转角和轨迹到jsonl文件"""
        with open(self.output_path, 'w') as f:
            for entry in processed:
                f.write(json.dumps(entry) + '\n')
        
    def normalize_angle_neg_pi_to_pi(self, angle):
        """将角度归一化到 [-π, π] 范围。查看了一下结果，本身就已经都是合理的了"""
        import math
        angle = angle % (2 * math.pi)  # 先归一化到 [0, 2π]
        if angle > math.pi:
            angle -= 2 * math.pi
        return angle

if __name__ == "__main__":
    # 创建PromptKittiPlanning类的实例并调用处理流程
    kitti_planning = PromptKittiPlanning()
    kitti_planning.process()
    print("处理完成，轨迹已保存。")