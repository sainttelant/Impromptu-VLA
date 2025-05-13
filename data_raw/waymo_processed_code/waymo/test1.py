import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

def extract_ego_car_coordinates(segment_path):
    dataset = tf.data.TFRecordDataset(segment_path, compression_type='')
    for idx, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        # 获取自车的坐标系信息
        ego_pose = frame.pose.transform
        print(f"Frame {idx}: Ego Car Pose: {ego_pose}")
        # ego_pose是齐次变换矩阵的结构
        自车的坐标系可以通过pose字段中的transform矩阵获取，该矩阵将传感器数据从局部坐标系转换到全局坐标系。
        Frame 196: Ego Car Pose: [0.624343064979528, 0.7810356217861137, -0.013382627288671213, -106.8654367383293, -0.781012536469265, 0.6238196717323613, -0.029469221869159026, 15191.579480229337, -0.014668165863996148, 0.028850903987700387, 0.9994760978879278, -28.57, 0.0, 0.0, 0.0, 1.0]
# 旋转部分（3×3矩阵）：
# 第一行：[0.6247939807537802, 0.7806679579800881, -0.013784810373889008]
# 第二行：[-0.7806554389449276, 0.6242618811997943, -0.029566692858621166]
# 第三行：[-0.014476438082177853, 0.029234278922063203, 0.9994677531948492]
# 这部分描述了自车坐标系相对于全局坐标系的方向（即自车的朝向）。
# 平移部分（最后一列）：
# [-106.81076364957008, 15191.51090219697, -28.572]
# 这部分描述了自车在全局坐标系中的位置：
# X轴方向的平移：-106.81076364957008
# Y轴方向的平移：15191.51090219697
# Z轴方向的平移：-28.572
# 示例：处理一个数据段
segment_path = "/DATA_EDS2/shenlc2403/feedfwd_driving/data/waymo/validation/segment-30779396576054160_1880_000_1900_000_with_camera_labels.tfrecord"
extract_ego_car_coordinates(segment_path)


# import numpy as np

# # 将矩阵转换为numpy数组
# ego_pose_matrix = np.array([
#     [0.6247939807537802, 0.7806679579800881, -0.013784810373889008, -106.81076364957008],
#     [-0.7806554389449276, 0.6242618811997943, -0.029566692858621166, 15191.51090219697],
#     [-0.014476438082177853, 0.029234278922063203, 0.9994677531948492, -28.572],
#     [0, 0, 0, 1]
# ])

# # 计算逆矩阵
# inverse_ego_pose_matrix = np.linalg.inv(ego_pose_matrix)
# print("Inverse Ego Car Pose Matrix:\n", inverse_ego_pose_matrix)