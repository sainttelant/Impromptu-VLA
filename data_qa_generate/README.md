Here's a translated and reformatted version of your README content:

-----

# File Organization Overview

## Folders

### **data**

This folder is used for editing and generating `dataset_info.json`. Please refer to `data/README.md` for detailed information.

### **data\_engine**

Within this folder, you only need to focus on the `datasets` subfolder. It contains `navsim` and `nuscenes` subfolders, which hold the code for evaluation on these two datasets. For evaluation on **nuScenes**, you can run:

```bash
nuscenes/scripts/evaluation_nuscenes.py --mode x-y --jsonl_file /path/to/your/result.jsonl --output_file /path/to/save.json
```

Please note that the evaluation for **Navsim** is incomplete and has not been thoroughly tested. You can ignore this part of the code as we do not guarantee its correctness.

### **data\_traj\_generate**

This folder contains scripts for extracting trajectory data from various datasets, organizing and computing it into end-to-end trajectory prediction QA data, and filtering out unreasonable trajectory data.

## Files

The remaining `xxx_qa.py` files are used to generate corresponding QA data for various datasets.

-----