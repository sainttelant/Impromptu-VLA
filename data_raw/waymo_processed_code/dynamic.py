import os
import json
import numpy as np
import pdb
import pickle
from tqdm import tqdm
import argparse

def numpy_array_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() 
    
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def trans(folder):
    with open(f"{folder}/scenario.pt", "rb") as f:
        data = pickle.load(f)
    
    dynamics = {}
    for k, v in data['metas']['dynamic_stats'].items():
        if v['n_dynamic'] == 0:
            continue
        if k == 'unknown':
            continue
        if k not in dynamics:
            dynamics[k] = []
        for idx in v['is_dynamic']:
            dynamics[k].append(data['objects'][idx])

    with open(f"{folder}/dynamic_objects.json", "w") as f:
        json.dump(dynamics, f, default=numpy_array_serializer, indent=4)
        
    for k, v in dynamics.items():
        for obj in v:
            for seg in obj["segments"]:
                seg["data"]["transform"] = [[trans[0][-1], trans[1][-1], trans[2][-1], trans[3][-1]] for trans in seg["data"]["transform"]]
    with open(f"{folder}/dynamic_objects_pose.json", "w") as f:
        json.dump(dynamics, f, default=numpy_array_serializer, indent=4)  

def main():
    parser = argparse.ArgumentParser(description='Extract dynamic objects from Waymo dataset')
    parser.add_argument(
        '--root', 
        type=str, 
        default='Impromptu-VLA/data_raw/waymo_processed',
        help='Root directory of processed Waymo data'
    )
    args = parser.parse_args()
    
    root = args.root
    
    for split in sorted(os.listdir(root)):
        print(f"Processing {split}...")
        for segment in tqdm(sorted(os.listdir(f"{root}/{split}"))):
            folder = f"{root}/{split}/{segment}"
            trans(folder)

if __name__ == "__main__":
    main()