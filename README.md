# Impromptu-VLA

Our dataset can be accessed at [huggingface](https://huggingface.co/datasets/aaaaaap/unstructed)

If you want to create our benchmark QA data from scratch:

1. First, organize the data download based on `data_raw`.
2. Parse the data according to the code and instructions in the folder (for the `waymo` and `mapillary_sls` datasets).
3. Enter the main directory.Create a symbolic link for `navsim`:
   ```bash
   ln -s /data_raw/navsim /data_qa_generate/data_engine/data_storage/external_datasets/navsim
   ```
4. After the data is successfully organized, run the following script:
   ```bash
   bash scripts/data_qa_generate.sh
   ```
---
### ✨ Environment Configuration

We leverage some powerful open-source libraries to make this project shine. To ensure a smooth experience, please configure your environment by referring to their official documentation.

Here are the key players:

* **sglang**: Your go-to for efficient large language model serving. Check out their setup guide here: [sglang](https://github.com/sgl-project/sglang) ✨
* **LLaMA-Factory**: A comprehensive and user-friendly framework for fine-tuning large language models. Dive into their documentation for installation details: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 🛠️
* **vLLM**: For high-throughput and low-latency inference. Find out how to get it running here: [vllm](https://github.com/vllm-project/vllm) ⚡

**Pro Tip:** We highly recommend creating a dedicated virtual environment (using tools like `conda` or `venv`) to manage the dependencies for this project. This helps keep your workspace clean and avoids conflicts with other Python projects. Happy configuring! 👩‍💻

### Download Pre-trained Models

|Method|Download|
|-|-|
|  3B Base+nuScenes  | [HF Hub](https://huggingface.co/aaaaaap/ImpromptuVLAModel/tree/main/3B_Base_finetune) |
|   3B Base+Impromptu   | [HF Hub](https://huggingface.co/aaaaaap/ImpromptuVLAModel/tree/main/3B_AD) |
|   3B Base+Impromptu+nuScenes   | [HF Hub](https://huggingface.co/aaaaaap/ImpromptuVLAModel/tree/main/3B_AD_finetune) |
|   7B Base+nuScenes   | [HF Hub](https://huggingface.co/aaaaaap/ImpromptuVLAModel/tree/main/7B_Base_finetune) |
|    7B Base+Impromptu  | [HF Hub](https://huggingface.co/aaaaaap/ImpromptuVLAModel/tree/main/7B_AD) |
|   7B Base+Impromptu+nuScenes   | [HF Hub](https://huggingface.co/aaaaaap/ImpromptuVLAModel/tree/main/7B_AD_finetune) |

#### Open-loop Evaluation

| Method                                     | L2 Error (m) $\downarrow$ (1s) | L2 Error (m) $\downarrow$ (2s) | L2 Error (m) $\downarrow$ (3s) | L2 Error (m) $\downarrow$ (Avg.) |
|--------------------------------------------|--------------------------------|--------------------------------|--------------------------------|----------------------------------|
| DriveVLM      | 0.18                           | 0.34                           | 0.68                           | 0.40                             |
| OmniDrive    | 0.14                           | 0.29                           | 0.55                           | 0.33                             |
| DriveVLM-Dual  | 0.15                           | 0.29                           | **0.48** | 0.31                             |
| EMMA (random init) | 0.15                           | 0.33                           | 0.63                           | 0.37                             |
| EMMA            | 0.14                           | 0.29                           | 0.54                           | 0.32                             |
| ~~EMMA~~+             | ~~0.13~~                           | ~~0.27~~                           | ~~0.48~~                           | ~~0.29~~                             |
| 3B Base+nuScenes                           | 0.14                           | 0.30                           | 0.58                           | 0.34                             |
| 3B Base+Impromptu+nuScenes                 | **0.13** | **0.27** | 0.52                           | **0.30** |
| 7B Base+nuScenes                           | **0.13** | 0.28                           | 0.55                           | 0.32                             |
| 7B Base+Impromptu+nuScenes                 | **0.13** | **0.27** | 0.53                           | **0.30** |

#### Close-loop Evaluation

| Source                    | Method                       | NeuroNCAP Score $\uparrow$ (Avg.) | NeuroNCAP Score $\uparrow$ (Stat.) | NeuroNCAP Score $\uparrow$ (Frontal) | NeuroNCAP Score $\uparrow$ (Side) | Collision rate (%) $\downarrow$ (Avg.) | Collision rate (%) $\downarrow$ (Stat.) | Collision rate (%) $\downarrow$ (Frontal) | Collision rate (%) $\downarrow$ (Side) |
|---------------------------|------------------------------|-----------------------------------|------------------------------------|--------------------------------------|-----------------------------------|----------------------------------------|-----------------------------------------|-------------------------------------------|----------------------------------------|
| CVPR 2023                 | UniAD                   | 0.73                              | 0.84                               | 0.10                                 | 1.26                              | 88.6                                   | 87.8                                    | 98.4                                      | 79.6                                   |
| ICCV 2023                 | VAD                      | 0.66                              | 0.47                               | 0.04                                 | 1.45                              | 92.5                                   | 96.2                                    | 99.6                                      | 81.6                                   |
| ICRA 2025                 | SparseDrive             | 0.92                              | -                                  | -                                    | -                                 | 93.9                                   | -                                       | -                                         | -                                      |
| CVPR 2025                 | BridgeAD-S               | 1.52                              | -                                  | -                                    | -                                 | 76.2                                   | -                                       | -                                         | -                                      |
| CVPR 2025                 | BridgeAD-B             | 1.60                              | -                                  | -                                    | -                                 | 72.6                                   | -                                       | -                                         | -                                      |
| -                         | 3B Base+nuScenes                | 1.77                              | **1.80** | 1.67                                 | 1.75                              | 72.5                                   | **68.0** | 73.0                                      | 71.5                                   |
| -                         | **3B Base+Impromptu+nuScenes**| **2.15** | 1.77                               | **2.31** | **2.10** | **65.5** | 70.0                                    | **59.0** | **65.0** |

### 🚀 Model Training

To start training, simply run the following command:

```bash
llamafactory-cli train <yaml_path>
```

Replace `<yaml_path>` with the path to your training configuration file. For example:

```bash
llamafactory-cli train train/Qwen2_5-VL/QA_train_sub_fin_nu/3B_full_QA_train_bs8.yaml
```

This command will launch the training process based on the settings specified in your YAML config file. Make sure the path is correct and all necessary parameters are properly configured.


### 🧠 Inference

To run inference with a fine-tuned model, you need to use the following command:

```bash
llamafactory-cli export \
  --model_name_or_path <path_to_base_model> \
  --adapter_name_or_path <path_to_lora_adapter_checkpoint> \
  --template qwen2_vl \
  --finetuning_type lora \
  --export_dir <path_to_save_merged_model> \
  --cutoff_len 4096 \
  --export_size 2 \
  --export_device cpu \
  --export_legacy_format false
```

Replace the placeholders with your actual paths:

* `<path_to_base_model>`: Path to the original pretrained model (e.g., Qwen2-VL-3B-Instruct)
* `<path_to_lora_adapter_checkpoint>`: Path to the fine-tuned LoRA checkpoint (e.g., `checkpoint-xxx`)
* `<path_to_save_merged_model>`: Directory to save the merged model

### Prompts
The prompts we use can be found in [prompts](prompts.md).

### 📊 Close-loop Evaluation with NeuroNCAP

To understand the system's performance within a closed-loop simulation environment, delve into the specifics of our NeuroNCAP-based evaluation: [Close-loop Evaluation](neuroncap_evaluation/evaluation.md) 🎮
