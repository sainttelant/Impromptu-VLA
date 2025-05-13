# 需要先对mapl_sls数据集进行软连接处理，代码中使用的是带软连接的数据，而且更加方便查看统一序列数据集
import os
import pandas as pd
from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
print(project_root)

base_dir = project_root / "data_raw" / "mapillary_sls" / "train_val"
out_dir = project_root  / "data_raw" / "mapillary_sls" / "train_val"
# base_dir = project_root / "data_raw" / "mapillary_sls" / "test"
# out_dir = project_root  / "data_raw" / "mapillary_sls" / "test"

def process_dataset():
    # 获取所有城市目录并排序
    city_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
        
    for city_dir in city_dirs:  
        print(f"Processing city: {city_dir.name}")
        for data_type in ['query']:
            type_path = city_dir / data_type
            if not type_path.exists():
                print(f"Skipping missing data type: {type_path}")
                continue
            
            # 路径设置
            seq_info_path = type_path / "seq_info.csv"
            images_dir = type_path / "images"
            simages_dir = out_dir / city_dir.name / data_type / "simages"  # 修复路径拼接问题
            
            # 确保 simages_dir 存在
            try:
                simages_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                print(f"Permission denied: Unable to create directory {simages_dir}")
                continue
            
            if not seq_info_path.exists():
                print(f"Missing seq_info.csv in {type_path}")
                continue
            
            # 读取序列信息
            try:
                df = pd.read_csv(seq_info_path)
            except Exception as e:
                print(f"Error reading {seq_info_path}: {str(e)}")
                continue
            
            # 按 sequence_key 分组处理
            for seq_key, seq_group in df.groupby('sequence_key'):
                # 按帧号排序并分割子场景
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
                
                # 处理每个子场景
                for scene in sub_scenes:
                    if not scene:
                        continue
                    
                    # 生成场景ID
                    start_frame = scene[0]['frame_number']
                    scene_id = f"{seq_key}-{start_frame}"
                 
                    # 创建场景目录
                    scene_dir = simages_dir / scene_id
                    try:
                        scene_dir.mkdir(parents=True, exist_ok=True)
                    except PermissionError:
                        print(f"Permission denied: Unable to create directory {scene_dir}")
                        continue
                    
                    # 创建软链接
                    for idx, record in enumerate(scene):
                        src_path = images_dir / f"{record['key']}.jpg"
                        if not src_path.exists():
                            print(f"Missing image: {src_path}")
                            continue
                        
                        # 生成目标文件名（三位数序号）
                        dest_name = f"{idx:03d}.jpg"
                        dest_path = scene_dir / dest_name
                        
                        if not dest_path.exists():
                            try:
                                os.symlink(src_path.resolve(), dest_path)
                            except FileExistsError:
                                pass  # 忽略已存在的链接
                            except Exception as e:
                                print(f"Error creating symlink {dest_path}: {str(e)}")

if __name__ == "__main__":
    process_dataset()