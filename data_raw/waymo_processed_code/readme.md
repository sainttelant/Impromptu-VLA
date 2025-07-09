The processing code is from an open-source project:  
[https://github.com/PJLab-ADG/neuralsim/tree/faba099e0feb11ea0089490a5e87565e25bc4a2c/dataio/autonomous_driving/waymo](https://github.com/PJLab-ADG/neuralsim/tree/faba099e0feb11ea0089490a5e87565e25bc4a2c/dataio/autonomous_driving/waymo)  
The code already contains attribution, you can directly follow this project.

Below is our execution method for reference:  
**Please pay attention to version compatibility**  
We used:  
- CUDA 12.4  
- waymo-open-dataset-tf-2-11-0==1.6.1  

### Training set:
```bash
python waymo_parse_tfrecord_main.py \
    --root Impromptu-VLA/data_raw/waymo/training \
    --out_root Impromptu-VLA/data_raw/waymo_processed/training \
    -j16 \
    --ignore_existing
```

### Validation set:
```bash
python waymo_parse_tfrecord_main.py \
    --root Impromptu-VLA/data_raw/waymo/validation \
    --out_root Impromptu-VLA/data_raw/waymo_processed/validation \
    -j16 \
    --ignore_existing
```

Running the above command will convert the original data from tfrecord format to a more readable form. The important files to use are the `images` folder and `scenario.pt` (which stores scene data). You can then run `python data_raw/waymo_processed_code/dynamic.py --root Impromptu-VLA/data_raw/waymo_processed`, which will extract the required dynamic object data from `scenario.pt` and generate `dynamic_objects.json` and `dynamic_objects_pose.json` to store the original transform matrices of dynamic objects and the x, y, z coordinates extracted from these matrices, respectively. It will also save `raw_poses.npy`, `base_pose.npy`, and `re_poses.npy`, which are the transform matrices, the first element matrix of the sequence, and the inverse of the original transform matrices, respectively. These will be used when generating QA in `data_qa_generate/waymo_qa.py`.

Note:
You need to keep the tfrecord files, as they will be used by:
Impromptu-VLA/data_qa_generate/data_traj_generate/pipeline_waymo_traj.py
Alternatively, you can modify the code logic in this file.
