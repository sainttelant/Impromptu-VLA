"""
@file   preprocess.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Waymo dataset preprocess.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
from glob import glob

from waymo_open_dataset import dataset_pb2

from waymo.filter_dynamic import collect_loc_motion


def set_env(depth: int):
    # Add project root to sys.path
    current_file_path = os.path.abspath(__file__)
    project_root_path = os.path.dirname(current_file_path)
    for _ in range(depth):
        project_root_path = os.path.dirname(project_root_path)
    if project_root_path not in sys.path:
        sys.path.append(project_root_path)
        print(f"Added {project_root_path} to sys.path")


set_env(3)

import functools  # noqa: E402
import io  # noqa: E402
import pickle  # noqa: E402
from typing import List  # noqa: E402

import numpy as np  # noqa: E402

# from dataio.autonomous_driving.waymo.waymo_dataset import *
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

WAYMO_CLASSES = ["unknown", "Vehicle", "Pedestrian", "Sign", "Cyclist"]
WAYMO_CAMERAS = [
    "FRONT",
    "FRONT_LEFT",
    "SIDE_LEFT",
    "FRONT_RIGHT",
    "SIDE_RIGHT",
]  # NOTE: name order in frame.images
WAYMO_LIDARS = [
    "TOP",
    "FRONT",
    "SIDE_LEFT",
    "SIDE_RIGHT",
    "REAR",
]  # NOTE: name order in frame.lasers


def idx_to_camera_id(camera_index):
    # return f'camera_{camera_index}'
    return f"camera_{WAYMO_CAMERAS[camera_index]}"


def idx_to_frame_str(frame_index):
    return f"{frame_index:08d}"


def idx_to_img_filename(frame_index):
    return f"{idx_to_frame_str(frame_index)}.jpg"


def idx_to_mask_filename(frame_index, compress=True):
    ext = "npz" if compress else "npy"
    return f"{idx_to_frame_str(frame_index)}.{ext}"


def idx_to_lidar_id(lidar_index):
    # return f'lidar_{lidar_index}'
    return f"lidar_{WAYMO_LIDARS[lidar_index]}"


def idx_to_lidar_filename(frame_index):
    return f"{idx_to_frame_str(frame_index)}.npz"


def file_to_scene_id(fname):
    return os.path.splitext(os.path.basename(os.path.normpath(fname)))[0]


def parse_seq_file_list(
    root: str, seq_list_fpath: str = None, seq_list: List[str] = None
):
    assert os.path.exists(root), f"Not exist: {root}"

    if seq_list is None and seq_list_fpath is not None:
        with open(seq_list_fpath, "r") as f:
            seq_list = f.read().splitlines()

    if seq_list is not None:
        seq_list = [s.split(",")[0].rstrip(".tfrecord") for s in seq_list]
        seq_fpath_list = []
        for s in seq_list:
            seq_fpath = os.path.join(root, f"{s}.tfrecord")
            assert os.path.exists(seq_fpath), f"Not exist: {seq_fpath}"
            seq_fpath_list.append(seq_fpath)
    else:
        seq_fpath_list = list(sorted(glob(os.path.join(root, "*.tfrecord"))))

    assert len(seq_fpath_list) > 0, f"No matching .tfrecord found in: {root}"
    return seq_fpath_list


# def collect_loc_motion(scenario: dict, loc_eps=0.03):
#     """
#     return path: {id: path_xyz}
#     path_xyz: path_x: [] (3*length)
#     """
#     categ_stats = {}
#     for oid, odict in scenario["objects"].items():
#         class_name = odict["class_name"]
#         # Location at world coordinate
#         loc_diff_norms = []
#         for seg in odict["segments"]:
#             locations = seg["data"]["transform"][:, :3, 3]
#             loc_diff = np.diff(locations, axis=0)
#             loc_diff_norm = np.linalg.norm(loc_diff, axis=-1)
#             loc_diff_norms.append(loc_diff_norm)
#         categ_stats.setdefault(class_name, {})[oid] = loc_diff_norms
#     return categ_stats


def collect_box_speed(dataset):
    """
    initial version: xiaohang
    filter the bbox based on speed
    input: one label
    output: True, moving; False, static
    """

    categories = {}
    for frame in dataset:
        for label in frame.laser_labels:
            class_name = WAYMO_CLASSES[int(label.type)]
            if class_name not in categories:
                categories[class_name] = {}

            meta = label.metadata
            if label.id not in categories[class_name]:
                categories[class_name][label.id] = dict(motions=[])

            categories[class_name][label.id]["motions"].append(
                np.linalg.norm([meta.speed_x, meta.speed_y])
            )

    return categories


def stat_dynamic_objects(dataset, speed_eps=0.2, loc_eps=0.03):
    # from .waymo_dataset import WAYMO_CLASSES

    stats = {
        cls_name: {"n_dynamic": 0, "is_dynamic": [], "by_speed": [], "by_loc": []}
        for cls_name in WAYMO_CLASSES
    }
    # ------------------------------------------------
    # Filter according to speed_x and speed_y
    speed_stats = collect_box_speed(dataset)
    for cls_name, cls_dict in speed_stats.items():
        by_speed = []
        for str_id, item_dict in cls_dict.items():
            if np.array(item_dict["motions"]).max() > speed_eps:
                by_speed.append(str_id)
        stats[cls_name]["by_speed"] = by_speed
    # ------------------------------------------------
    # Filter according to center_x and center_y
    loc_motion_stats, _ = collect_loc_motion(dataset)
    for cls_name, cls_dict in loc_motion_stats.items():
        by_loc = []
        for str_id, item_dict in cls_dict.items():
            if np.array(item_dict["motions"]).max() > loc_eps:
                by_loc.append(str_id)
        stats[cls_name]["by_loc"] = by_loc
    # ------------------------------------------------
    # Collect results from box_speed and loc_motion
    for cls_name, cls_dict in stats.items():
        li_dyna = list(set(cls_dict["by_speed"]) | set(cls_dict["by_loc"]))
        stats[cls_name]["is_dynamic"] = li_dyna
        stats[cls_name]["n_dynamic"] = len(li_dyna)

    return stats


def process_single_sequence(
    sequence_file: str,
    out_root: str,
    rgb_dirname: str = None,
    lidar_dirname: str = None,
    pcl_dirname: str = None,
    mask_dirname: str = None,
    # Other configs
    class_names: List[str] = WAYMO_CLASSES,
    should_offset_pos=True,
    should_offset_timestamp=True,
    should_process_gt=True,
    ignore_existing=False,
):
    # NOTE:
    # 1. It seems that tensorflow==2.11 is no longer thread safe (compared to tf==2.6.0);
    #   Using multi-threading causes tons of errors randomly everywhere !!! TAT
    # 2. Hence, we need to use multi-processing instead of multi-threading;
    #   In this case, we need to import tensorflow (and any module that will import tensorflow inside)
    #       in process function instead of globally, to prevent CUDA initialization BUG.
    #   Multi-processing consumes more GPU mem even with set_memory_growth, compared to multi-threading.

    # NOTE: For tensorflow>=2.2 (2.11.0 currently)
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.utils import frame_utils, range_image_utils, transform_utils

    from waymo.waymo_filereader import WaymoDataFileReader

    if not os.path.exists(sequence_file):
        print(f"Not exist: {sequence_file}")
        return

    try:
        # dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
        dataset = WaymoDataFileReader(str(sequence_file))

        scene_objects = dict()
        scene_observers = dict()

        # ---- Use frame0 to process some meta info
        # frame0_data = bytearray(next(iter(dataset)).numpy())
        # frame0 = dataset_pb2.Frame()
        # frame0.ParseFromString(frame0_data)
        frame0 = next(iter(dataset))

        # scene_id = frame0.context.name
        scene_id = file_to_scene_id(sequence_file)

        # 获取指定场景ID对应的文件名（行号）
        import os
        current_dir = os.path.dirname(__file__)
        relative_path = os.path.join(current_dir, 'waymo', 'val.txt')
        total_list = open(relative_path, "r").readlines()  
        # relative_path = os.path.join(current_dir, 'waymo', 'train.txt')
        # total_list = open(relative_path, "r").readlines()  
        #attention： 这个txt的作用是对scene_id 的重映射，但我没找到准确的txt，导致一个都对不上，所以scene_id 将保持它最初的值，无论什么txt都不影响文件夹数目
        for index, line in enumerate(total_list):
            if scene_id == line.strip():
                scene_id = f"{index:03d}"
                break
        # ---- Outputs
        os.makedirs(os.path.join(out_root, scene_id), exist_ok=True)

        rgb_dir = os.path.join(out_root, scene_id, rgb_dirname) if rgb_dirname else None
        lidar_dir = (
            os.path.join(out_root, scene_id, lidar_dirname) if lidar_dirname else None
        )
        pcl_dir = os.path.join(out_root, scene_id, pcl_dirname) if pcl_dirname else None
        scenario_fpath = os.path.join(out_root, scene_id, "scenario.pt")
        if ignore_existing:
            if (rgb_dir is None) or os.path.exists(rgb_dir):
                rgb_dir = None
            if (lidar_dir is None) or os.path.exists(lidar_dir):
                lidar_dir = None
            if (pcl_dir is None) or os.path.exists(pcl_dir):
                pcl_dir = None

        # NOTE: To normalize segments poses (for letting x=0,y=0,z=0 @ 0-th frame)
        world_offset = np.zeros(
            [
                3,
            ]
        )
        if should_offset_pos:
            # ---- OPTION1: Use the camera_0's 0-th pose as offset
            # extr00 = np.array(frame0.context.camera_calibrations[0].extrinsic.transform).reshape(4,4)
            # pose00 = np.array(frame0.images[0].pose.transform).reshape(4,4)
            # c2w00 = pose00 @ extr00
            # world_offset = c2w00[:3, 3]

            # ---- OPTION2: Use the vehicle's 0-th pose as offset (for waymo, the same with OPTION1: waymo's frame.pose is exactly camera0's pose)
            frame0_pose = np.array(frame0.pose.transform, copy=True).reshape(4, 4)
            world_offset = frame0_pose[:3, 3]
        timestamp_offset = 0
        if should_offset_timestamp:
            timestamp_offset = frame0.timestamp_micros / 1e6

        frame_timestamps = []

        # ------------------------------------------------------
        # --------    Dynamic object statistics     ------------
        # ------------------------------------------------------
        dynamic_stats = stat_dynamic_objects(dataset)

        # NOTE: Not used.
        # frame_inds_with_panoptic_label = []
        # for frame_ind, frame in enumerate(dataset):
        #     if frame.images[0].camera_segmentation_label.panoptic_label:
        #         frame_inds_with_panoptic_label.append(frame_ind)

        # --------------- per-frame processing
        # for frame_ind, framd_data in enumerate(dataset):
        # frame = dataset_pb2.Frame()
        # frame.ParseFromString(bytearray(framd_data.numpy()))

        # annotation.json and transform.json
        segment_name = None
        segment_out_dir = None
        sensor_params = None
        camera_frames = []
        lidar_frames = []
        annotations = []

        for frame_ind, frame in enumerate(tqdm(dataset, "processing...")):
            # ---- Ego pose
            frame_pose = np.array(frame.pose.transform, copy=True).reshape(4, 4)
            frame_pose[:3, 3] -= world_offset
            frame_timestamp = frame.timestamp_micros / 1e6
            if should_offset_timestamp:
                frame_timestamp -= timestamp_offset
            frame_timestamps.append(frame_timestamp)

            # ------------------------------------------------------
            # --------------     Frame Observers      --------------
            # ------------------------------------------------------
            if "ego_car" not in scene_observers:
                scene_observers["ego_car"] = dict(
                    class_name="EgoVehicle",
                    n_frames=0,
                    data=dict(v2w=[], global_timestamps=[], global_frame_inds=[]),
                )
            scene_observers["ego_car"]["n_frames"] += 1
            scene_observers["ego_car"]["data"]["v2w"].append(frame_pose)
            scene_observers["ego_car"]["data"]["global_timestamps"].append(
                frame_timestamp
            )
            scene_observers["ego_car"]["data"]["global_frame_inds"].append(frame_ind)

            # ------------------------------------------------------
            # ------------------     Cameras      ------------------
            # NOTE: !!! Waymo's images order is not 12345 !!!
            # frame.context.camera_calibrations[0,1,2,3,4].name:[1,2,3,4,5]
            # frame.images[0,1,2,3,4].name:                     [1,2,4,3,5]
            for j in range(len(WAYMO_CAMERAS)):
                c = frame.context.camera_calibrations[j]
                for _j in range(len(frame.images)):
                    if frame.images[_j].name == c.name:
                        break
                camera = frame.images[_j]
                assert c.name == camera.name == (j + 1)
                str_id = idx_to_camera_id(_j)

                camera_timestamp = camera.pose_timestamp
                if should_offset_timestamp:
                    camera_timestamp -= timestamp_offset

                h, w = c.height, c.width

                # fx, fy, cx, cy, k1, k2, p1, p2, k3
                fx, fy, cx, cy, *distortion = np.array(c.intrinsic)
                distortion = np.array(distortion)
                intr = np.eye(3)
                intr[0, 0] = fx
                intr[1, 1] = fy
                intr[0, 2] = cx
                intr[1, 2] = cy

                """
                    < opencv / colmap convention >                 --->>>   < waymo convention >
                    facing [+z] direction, x right, y downwards    --->>>  facing [+x] direction, z upwards, y left
                                z                                          z ↑ 
                              ↗                                              |  ↗ x
                             /                                               | /
                            /                                                |/
                            o------> x                              ←--------o
                            |                                      y
                            |
                            |
                            ↓ 
                            y
                """
                # NOTE: Opencv camera to waymo camera
                opencv_to_waymo = np.eye(4)
                opencv_to_waymo[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

                # NOTE: Waymo: extrinsic=[camera to vehicle]
                c2v = np.array(c.extrinsic.transform).reshape(4, 4)
                # NOTE: Waymo: pose=[vehicle to ENU(world)]
                v2w = np.array(camera.pose.transform).reshape(4, 4)
                v2w[:3, 3] -= world_offset
                # NOTE: [camera to ENU(world)]
                c2w = v2w @ c2v @ opencv_to_waymo

                if str_id not in scene_observers:
                    scene_observers[str_id] = dict(
                        class_name="Camera",
                        n_frames=0,
                        data=dict(
                            hw=[],
                            intr=[],
                            distortion=[],
                            c2v_0=[],
                            c2v=[],
                            sensor_v2w=[],
                            c2w=[],
                            global_timestamps=[],
                            global_frame_inds=[],
                        ),
                    )
                scene_observers[str_id]["n_frames"] += 1
                scene_observers[str_id]["data"]["hw"].append((h, w))
                scene_observers[str_id]["data"]["intr"].append(intr)
                scene_observers[str_id]["data"]["distortion"].append(distortion)
                scene_observers[str_id]["data"]["c2v_0"].append(c2v)
                scene_observers[str_id]["data"]["c2v"].append(c2v @ opencv_to_waymo)
                scene_observers[str_id]["data"]["sensor_v2w"].append(
                    v2w
                )  # v2w at each camera's timestamp
                scene_observers[str_id]["data"]["c2w"].append(c2w)
                scene_observers[str_id]["data"]["global_timestamps"].append(
                    camera_timestamp
                )
                scene_observers[str_id]["data"]["global_frame_inds"].append(frame_ind)

                # -------- Process observation groundtruths
                if should_process_gt and rgb_dir:
                    img = Image.open(io.BytesIO(camera.image))
                    assert [*(np.asarray(img)).shape[:2]] == [h, w]
                    img_cam_dir = os.path.join(rgb_dir, str_id)
                    os.makedirs(img_cam_dir, exist_ok=True)
                    img.save(os.path.join(img_cam_dir, idx_to_img_filename(frame_ind)))

            # ------------------------------------------------------
            # ---------------     Frame Objects      ---------------
            # ------------------------------------------------------
            for l in frame.laser_labels:
                str_id = str(l.id)
                # str_id = f"{scene_id}#{l.id}"

                if WAYMO_CLASSES[l.type] not in class_names:
                    continue

                if str_id not in scene_objects:
                    scene_objects[str_id] = dict(
                        id=l.id,
                        # class_ind=l.type,
                        class_name=WAYMO_CLASSES[l.type],
                        frame_annotations=[],
                    )

                # https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto
                box = l.box

                # Box coordinates in vehicle frame.
                tx, ty, tz = box.center_x, box.center_y, box.center_z

                # The heading of the bounding box (in radians).  The heading is the angle
                #   required to rotate +x to the surface normal of the box front face. It is
                #   normalized to [-pi, pi).
                c = np.math.cos(box.heading)
                s = np.math.sin(box.heading)

                # [object to vehicle]
                # https://github.com/gdlg/simple-waymo-open-dataset-reader/blob/d488196b3ded6574c32fad391467863b948dfd8e/simple_waymo_open_dataset_reader/utils.py#L32
                o2v = np.array(
                    [[c, -s, 0, tx], [s, c, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]]
                )

                # [object to ENU world]
                pose = frame_pose @ o2v

                # difficulty = l.detection_difficulty_level

                # tracking_difficulty = l.tracking_difficulty_level

                # Dimensions of the box. length: dim x. width: dim y. height: dim z.
                # length: dim_x: along heading; dim_y: verticle to heading; dim_z: verticle up
                dimension = [box.length, box.width, box.height]

                scene_objects[str_id]["frame_annotations"].append(
                    [[frame_ind, frame_timestamp], [pose, dimension]]
                )

        n_global_frames = frame_ind + 1

        # --------------- Per-observer processing
        for oid, odict in scene_observers.items():
            for k, v in odict["data"].items():
                odict["data"][k] = np.array(v)

        # --------------- Per-object processing: from frame annotations to frame attribute segments
        for oid, odict in scene_objects.items():
            obj_annos = odict.pop("frame_annotations")

            segments = []
            for i, ([frame_ind, frame_timestamp], [pose, dimension]) in enumerate(
                obj_annos
            ):
                if (i == 0) or (frame_ind - obj_annos[i - 1][0][0] != 1):
                    cur_segment = dict(
                        start_frame=frame_ind,
                        n_frames=None,
                        data=None,
                    )
                    cur_seg_data = dict(
                        transform=[],
                        scale=[],
                        global_timestamps=[],
                        global_frame_inds=[],
                    )

                # NOTE: Waymo assumes all annotations are captured at frame timestamp.
                cur_seg_data["global_timestamps"].append(frame_timestamp)

                cur_seg_data["transform"].append(pose)
                cur_seg_data["scale"].append(dimension)
                cur_seg_data["global_frame_inds"].append(frame_ind)

                if (i == len(obj_annos) - 1) or (
                    obj_annos[i + 1][0][0] - frame_ind != 1
                ):
                    # ----------------- Process last segment
                    for k, v in cur_seg_data.items():
                        cur_seg_data[k] = np.array(v)
                    cur_segment["n_frames"] = frame_ind - cur_segment["start_frame"] + 1
                    cur_segment["data"] = cur_seg_data
                    segments.append(cur_segment)

            odict["n_full_frames"] = n_global_frames
            odict["segments"] = segments

        scenario = dict()
        scenario["scene_id"] = scene_id
        scenario["metas"] = {
            "n_frames": n_global_frames,
            "world_offset": world_offset,
            "timestamp_offset": timestamp_offset,
            "frame_timestamps": np.array(frame_timestamps),
            "dynamic_stats": dynamic_stats,
        }
        scenario["objects"] = scene_objects
        scenario["observers"] = scene_observers

        with open(scenario_fpath, "wb") as f:
            pickle.dump(scenario, f)
            print(f"=> scenario saved to {scenario_fpath}")

    except Exception as e:
        print(f"Process waymo run into error: \n{e}")
        raise e

    return True


def create_dataset(
    root: str,
    seq_list_fpath: str,
    out_root: str,
    *,
    j: int = 1,
    should_offset_pos=True,
    should_process_gt=True,
    ignore_existing=False,
):
    # import concurrent.futures as futures

    from tqdm.contrib.concurrent import process_map, thread_map

    os.makedirs(out_root, exist_ok=True)

    seq_fpath_list = parse_seq_file_list(root, seq_list_fpath=seq_list_fpath)
    num_workers = min(j, len(seq_fpath_list))
    process_fn = functools.partial(
        process_single_sequence,
        out_root=out_root,
        rgb_dirname="images",
        lidar_dirname="lidars",
        pcl_dirname=None,
        should_offset_pos=should_offset_pos,
        should_process_gt=should_process_gt,
        ignore_existing=ignore_existing,
    )

    if num_workers == 1:
        for seq_fpath in tqdm(seq_fpath_list, "Processing waymo..."):
            process_fn(seq_fpath)
    else:
        process_map(
            process_fn, seq_fpath_list, max_workers=args.j, desc="Processing waymo..."
        )

        # with futures.ThreadPoolExecutor(num_workers) as executor:
        #     iterator = executor.map(process_fn, seq_fpath_list)
        #     next(iterator)


if __name__ == "__main__":
    """
    Usage:
        python preprocess.py --root /path/to/waymo/training --out_root /path/to/processed --seq_list /path/to/xxx.lst -j8
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="/mnt/ssd4t/waymo/buchong",
        required=False,
        help="Root directory of raw .tfrecords",
    )
    parser.add_argument(
        "--seq_list",
        type=str,
        default=None,
        help="Optional specify subset of sequences. If None, will process all sequences contained in args.root",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="/mnt/ssd4t/waymo_train/",
        required=False,
        help="Output root directory",
    )
    parser.add_argument("--no_offset_pose", action="store_true")
    parser.add_argument("--no_process_gt", action="store_true")
    parser.add_argument("--ignore_existing", action="store_true")
    parser.add_argument("-j", type=int, default=1, help="max num workers")
    args = parser.parse_args()
    create_dataset(
        args.root,
        args.seq_list,
        args.out_root,
        j=args.j,
        should_offset_pos=not args.no_offset_pose,
        should_process_gt=not args.no_process_gt,
        ignore_existing=args.ignore_existing,
    )
