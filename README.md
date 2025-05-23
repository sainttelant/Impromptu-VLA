# Impromptu-VLA

Our dataset can be accessed at https://huggingface.co/datasets/aaaaaap/unstructed

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

* **sglang**: Your go-to for efficient large language model serving. Check out their setup guide here: [https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) ✨
* **LLaMA-Factory**: A comprehensive and user-friendly framework for fine-tuning large language models. Dive into their documentation for installation details: [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 🛠️
* **vLLM**: For high-throughput and low-latency inference. Find out how to get it running here: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) ⚡

**Pro Tip:** We highly recommend creating a dedicated virtual environment (using tools like `conda` or `venv`) to manage the dependencies for this project. This helps keep your workspace clean and avoids conflicts with other Python projects. Happy configuring! 👩‍💻

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
