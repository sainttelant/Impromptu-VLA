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
    --root <your_waymo_download_path>/waymo/training \
    --out_root <your_project_path>/ImpromptuDriveBench/data_raw/waymo/training \
    -j16 \
    --ignore_existing
```

### Validation set:
```bash
python waymo_parse_tfrecord_main.py \
    --root <your_waymo_download_path>/waymo/validation \
    --out_root <your_project_path>/ImpromptuDriveBench/data_raw/waymo/validation \
    -j16 \
    --ignore_existing
```


Note:
You need to keep the tfrecord files, as they will be used by:
ImpromptuDriveBench/data_qa_generate/data_traj_generate/pipeline_waymo_planning.py
Alternatively, you can modify the code logic in this file.
