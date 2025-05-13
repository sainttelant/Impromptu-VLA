# 需要先对mapl_sls数据集进行软连接处理，代码中使用的是带软连接的数据，而且组织有序，更加方便查看场景序列
import os
import pandas as pd
from pathlib import Path

script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
print(project_root)

data_sets = {
    "train_val": project_root / "data_raw" / "mapillary_sls" / "train_val",
    "test": project_root / "data_raw" / "mapillary_sls" / "test"
}

def process_dataset(base_dir, out_dir):
    city_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
    for city_dir in city_dirs:
        print(f"Processing city: {city_dir.name}")
        for data_type in ['query']:
            type_path = city_dir / data_type
            if not type_path.exists():
                print(f"Skipping missing data type: {type_path}")
                continue
            seq_info_path = type_path / "seq_info.csv"
            images_dir = type_path / "images"
            simages_dir = out_dir / city_dir.name / data_type / "simages"
            try:
                simages_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                print(f"Permission denied: Unable to create directory {simages_dir}")
                continue
            if not seq_info_path.exists():
                print(f"Missing seq_info.csv in {type_path}")
                continue
            try:
                df = pd.read_csv(seq_info_path)
            except Exception as e:
                print(f"Error reading {seq_info_path}: {str(e)}")
                continue
            for seq_key, seq_group in df.groupby('sequence_key'):
                sorted_group = seq_group.sort_values('frame_number')
                sub_scenes = []
                current_scene = []
                prev_frame = None
                for _, row in sorted_group.iterrows():
                    current_frame = row['frame_number']
                    if prev_frame is None:
                        current_scene.append(row)
                    else:
                        if current_frame - prev_frame != 1:
                            sub_scenes.append(current_scene)
                            current_scene = [row]
                        else:
                            current_scene.append(row)
                    prev_frame = current_frame
                if current_scene:
                    sub_scenes.append(current_scene)
                for scene in sub_scenes:
                    if not scene:
                        continue
                    start_frame = scene[0]['frame_number']
                    scene_id = f"{seq_key}-{start_frame}"
                    scene_dir = simages_dir / scene_id
                    try:
                        scene_dir.mkdir(parents=True, exist_ok=True)
                    except PermissionError:
                        print(f"Permission denied: Unable to create directory {scene_dir}")
                        continue
                    for idx, record in enumerate(scene):
                        src_path = images_dir / f"{record['key']}.jpg"
                        if not src_path.exists():
                            print(f"Missing image: {src_path}")
                            continue
                        dest_name = f"{idx:03d}.jpg"
                        dest_path = scene_dir / dest_name
                        if not dest_path.exists():
                            try:
                                os.symlink(src_path.resolve(), dest_path)
                            except FileExistsError:
                                pass
                            except Exception as e:
                                print(f"Error creating symlink {dest_path}: {str(e)}")

if __name__ == "__main__":
    for mode in data_sets:
        print(f"Processing {mode} dataset...")
        base_dir = data_sets[mode]
        out_dir = data_sets[mode]
        process_dataset(base_dir, out_dir)