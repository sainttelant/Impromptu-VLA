# Impromptu-VLA

This repository contains the code for the following work:

> Impromptu VLA: Open Weights and Open Data for Driving Vision-Language-Action Models

## [ProjectPage](http://Impromptu-VLA.c7w.tech/)

Haohan Chi*,¹, Huan-ang Gao*,¹, Ziming Liu†,², Jianing Liu¹, Chenyu Liu¹, Jinwei Li¹, Kaisen Yang¹, Yangcheng Yu¹, Zeda Wang¹, Wenyi Li¹, Leichen Wang², Xingtao Hu², Hao Sun², Hang Zhao³, Hao Zhao¹,†

¹AIR, Tsinghua University, ²Bosch Research, ³IIIS, Tsinghua University, *Equal contribution, †Corresponding author

<div align="center">
   <img width="33%" src="images/tsinghua.png">
</div>

<br>
<div align="center">
  <img src="https://img.shields.io/github/license/ahydchh/Impromptu-VLA.svg" alt="License">
  <a href="https://arxiv.org/abs/2505.23757"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2505.23757-red"></a>
  <a href="https://huggingface.co/datasets/aaaaaap/unstructed"><img alt='Dataset' src="https://img.shields.io/badge/Dataset-Impromptu--VLA-blue"></a>
  <a href="https://Impromptu-VLA.c7w.tech/"><img alt='Project Page' src="https://img.shields.io/badge/Webpage-Impromptu--VLA-green"></a>
</div>
<br>

## Introductory Video
<video src="videos/intro.mp4" width="100%" style="max-width: 100%; height: auto;" controls loop muted></video>

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

<section class="section hero is-light">
    <div class="container is-max-desktop">
      <h2 class="title is-3 has-text-centered" style="margin-bottom: 1.2rem;">📊 Results</h2>
      <div style="max-width:900px; margin: 0 auto 2em auto; display: block;">
        <div style="overflow-x: auto;">
          <table class="table is-bordered is-striped is-narrow is-hoverable" style="font-size:0.95em; width:100%;">
            <caption style="caption-side:top; text-align:center; font-weight:bold; margin-bottom:0.5em;">
              Open-loop trajectory prediction L2 errors (m) on the nuScenes dataset.
            </caption>
            <thead>
              <tr>
                <th style="width:20%;">Method</th>
                <th style="width:20%;">1s</th>
                <th style="width:20%;">2s</th>
                <th style="width:20%;">3s</th>
                <th style="width:20%;">Avg.</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colspan="5"><em>Closed-source API-only Models</em></td>
              </tr>
              <tr>
                <td>GPT-4o<sup>1</sup></td>
                <td><b>0.28</b></td>
                <td><b>0.93</b></td>
                <td><b>2.02</b></td>
                <td style="background:#e3f0ff;"><b>1.07</b></td>
              </tr>
              <tr>
                <td>Claude-3.5-Sonnet<sup>1</sup></td>
                <td><u>0.29</u></td>
                <td>0.98</td>
                <td>2.12</td>
                <td style="background:#e3f0ff;">1.13</td>
              </tr>
              <tr>
                <td>Claude-3.7-Sonnet<sup>1</sup></td>
                <td><b>0.28</b></td>
                <td><u>0.94</u></td>
                <td><u>2.04</u></td>
                <td style="background:#e3f0ff;"><u>1.09</u></td>
              </tr>
              <tr>
                <td>Gemini-2.0-Flash<sup>1</sup></td>
                <td>0.31</td>
                <td>1.08</td>
                <td>2.36</td>
                <td style="background:#e3f0ff;">1.25</td>
              </tr>
              <tr>
                <td>Gemini-2.5-Pro<sup>1</sup></td>
                <td>0.37</td>
                <td>1.35</td>
                <td>2.96</td>
                <td style="background:#e3f0ff;">1.56</td>
              </tr>
              <tr>
                <td colspan="5"><em>Open-source Generalist VLMs</em></td>
              </tr>
              <tr>
                <td>LLaVA-1.6-Mistral-7B<sup>2</sup></td>
                <td>1.49</td>
                <td>3.38</td>
                <td>4.09</td>
                <td style="background:#e3f0ff;">2.98</td>
              </tr>
              <tr>
                <td>Llama-3.2-11B-Vision-Instruct<sup>2</sup></td>
                <td>1.54</td>
                <td>3.31</td>
                <td>3.91</td>
                <td style="background:#e3f0ff;">2.92</td>
              </tr>
              <tr>
                <td>Qwen2-VL-7B-Instruct<sup>2</sup></td>
                <td>1.45</td>
                <td>3.21</td>
                <td>3.76</td>
                <td style="background:#e3f0ff;">2.81</td>
              </tr>
              <tr>
                <td>DeepSeek-VL2-16B<sup>1</sup></td>
                <td>0.66</td>
                <td>1.68</td>
                <td>2.92</td>
                <td style="background:#e3f0ff;">1.75</td>
              </tr>
              <tr>
                <td>DeepSeek-VL2-28B<sup>1</sup></td>
                <td><b>0.37</b></td>
                <td><u>1.35</u></td>
                <td>2.96</td>
                <td style="background:#e3f0ff;">1.56</td>
              </tr>
              <tr>
                <td>LLaMA-3.2-11B-Vision-Instruct<sup>1</sup></td>
                <td>0.52</td>
                <td>1.42</td>
                <td><u>2.68</u></td>
                <td style="background:#e3f0ff;"><u>1.54</u></td>
              </tr>
              <tr>
                <td>LLaMA-3.2-90B-Vision-Instruct<sup>1</sup></td>
                <td>0.66</td>
                <td>1.71</td>
                <td>3.01</td>
                <td style="background:#e3f0ff;">1.79</td>
              </tr>
              <tr>
                <td>Qwen-2.5-VL-7B-Instruct<sup>1</sup></td>
                <td><u>0.46</u></td>
                <td><b>1.33</b></td>
                <td><b>2.55</b></td>
                <td style="background:#e3f0ff;"><b>1.45</b></td>
              </tr>
              <tr>
                <td colspan="5"><em>Training-based Driving Specialists (Existing Methods)</em></td>
              </tr>
              <tr>
                <td>UniAD<sup>3</sup></td>
                <td>0.42</td>
                <td>0.64</td>
                <td>0.91</td>
                <td style="background:#e3f0ff;">0.66</td>
              </tr>
              <tr>
                <td>VAD<sup>3</sup></td>
                <td>0.17</td>
                <td>0.34</td>
                <td>0.60</td>
                <td style="background:#e3f0ff;">0.37</td>
              </tr>
              <tr>
                <td>BEV-Planner<sup>3</sup></td>
                <td><u>0.16</u></td>
                <td><b>0.32</b></td>
                <td><b>0.57</b></td>
                <td style="background:#e3f0ff;"><b>0.35</b></td>
              </tr>
              <tr>
                <td>Ego-MLP<sup>3</sup>*</td>
                <td><b>0.15</b></td>
                <td><b>0.32</b></td>
                <td><u>0.59</u></td>
                <td style="background:#e3f0ff;"><b>0.35</b></td>
              </tr>
              <tr>
                <td colspan="5"><em>Ours and Key Competitors (Specialized Driving Models)</em></td>
              </tr>
              <tr>
                <td>DriveVLM<sup>3</sup></td>
                <td>0.18</td>
                <td>0.34</td>
                <td>0.68</td>
                <td style="background:#e3f0ff;">0.40</td>
              </tr>
              <tr>
                <td>OmniDrive<sup>3</sup></td>
                <td><u>0.14</u></td>
                <td>0.29</td>
                <td>0.55</td>
                <td style="background:#e3f0ff;">0.33</td>
              </tr>
              <tr>
                <td>DriveVLM-Dual<sup>3</sup></td>
                <td>0.15</td>
                <td>0.29</td>
                <td><b>0.48</b></td>
                <td style="background:#e3f0ff;">0.31</td>
              </tr>
              <tr>
                <td>EMMA (random init)<sup>3</sup></td>
                <td>0.15</td>
                <td>0.33</td>
                <td>0.63</td>
                <td style="background:#e3f0ff;">0.37</td>
              </tr>
              <tr>
                <td>EMMA<sup>3</sup></td>
                <td><u>0.14</u></td>
                <td>0.29</td>
                <td>0.54</td>
                <td style="background:#e3f0ff;">0.32</td>
              </tr>
              <tr style="color:lightgray;">
                <td>EMMA+<sup>3</sup></td>
                <td>0.13</td>
                <td>0.27</td>
                <td>0.48</td>
                <td style="background:#e3f0ff;">0.29</td>
              </tr>
              <tr>
                <td>3B Base+nuScenes</td>
                <td>0.14</td>
                <td>0.30</td>
                <td>0.58</td>
                <td style="background:#e3f0ff;">0.34</td>
              </tr>
              <tr>
                <td>3B Base+Impromptu+nuScenes</td>
                <td><b>0.13</b></td>
                <td><b>0.27</b></td>
                <td><u>0.52</u></td>
                <td style="background:#e3f0ff;"><b>0.30</b></td>
              </tr>
              <tr>
                <td>7B Base+nuScenes</td>
                <td><b>0.13</b></td>
                <td><u>0.28</u></td>
                <td>0.55</td>
                <td style="background:#e3f0ff;"><u>0.32</u></td>
              </tr>
              <tr>
                <td>7B Base+Impromptu+nuScenes</td>
                <td><b>0.13</b></td>
                <td><b>0.27</b></td>
                <td>0.53</td>
                <td style="background:#e3f0ff;"><b>0.30</b></td>
              </tr>
            </tbody>
          </table>
        </div>
        <div style="font-size:0.85em; color:#555; margin-top:0.5em; text-align:center;">
          <b>Note:</b> Best results within each category are in <b>bold</b>, second best are <u>underlined</u>.
          <sup>1</sup> from <a href="https://arxiv.org/abs/2505.00284" target="_blank"
            style="color: #1976d2;">LightEMMA</a>, <sup>2</sup> from <a href="https://arxiv.org/abs/2412.15208"
            target="_blank" style="color: #1976d2;">OpenEMMA</a>, <sup>3</sup> from <a
            href="https://arxiv.org/abs/2410.23262" target="_blank" style="color: #1976d2;">EMMA</a>.
        </div>
      </div>
  </section>
  <div style="max-width:900px; margin: 0 auto 2em auto; display: block;">
        <div style="overflow-x: auto;">
          <table class="table is-bordered is-striped is-narrow is-hoverable" style="font-size:0.95em; width:100%;">
            <caption style="caption-side:top; text-align:center; font-weight:bold; margin-bottom:0.5em;">
              Results on NeuroNCAP
            </caption>
            <thead>
              <tr>
                <th rowspan="2">Source</th>
                <th rowspan="2">Method</th>
                <th colspan="4">NeuroNCAP Score ↑</th>
                <th colspan="4">Collision rate (%) ↓</th>
              </tr>
              <tr>
                <th>Avg.</th>
                <th>Stat.</th>
                <th>Frontal</th>
                <th>Side</th>
                <th>Avg.</th>
                <th>Stat.</th>
                <th>Frontal</th>
                <th>Side</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>CVPR 2023</td>
                <td>UniAD<sup>2</sup></td>
                <td>0.73</td>
                <td>0.84</td>
                <td>0.10</td>
                <td>1.26</td>
                <td>88.6</td>
                <td>87.8</td>
                <td>98.4</td>
                <td>79.6</td>
              </tr>
              <tr>
                <td>ICCV 2023</td>
                <td>VAD<sup>2</sup></td>
                <td>0.66</td>
                <td>0.47</td>
                <td>0.04</td>
                <td>1.45</td>
                <td>92.5</td>
                <td>96.2</td>
                <td>99.6</td>
                <td>81.6</td>
              </tr>
              <tr>
                <td>ICRA 2025</td>
                <td>SparseDrive<sup>1</sup></td>
                <td>0.92</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>93.9</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
              </tr>
              <tr>
                <td>CVPR 2025</td>
                <td>BridgeAD-S<sup>1</sup></td>
                <td>1.52</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>76.2</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
              </tr>
              <tr>
                <td>CVPR 2025</td>
                <td>BridgeAD-B<sup>1</sup></td>
                <td>1.60</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>72.6</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
              </tr>
              <tr style="background:#f0f0f0;">
                <td>-</td>
                <td><u>Base+nuScenes</u></td>
                <td><u>1.77</u></td>
                <td><b>1.80</b></td>
                <td><u>1.67</u></td>
                <td><u>1.75</u></td>
                <td><u>72.5</u></td>
                <td><b>68.0</b></td>
                <td><u>73.0</u></td>
                <td><u>71.5</u></td>
              </tr>
              <tr style="background:#f0f0f0;">
                <td>-</td>
                <td><b>Base+Impromptu+nuScenes</b></td>
                <td><b>2.15</b></td>
                <td><u>1.77</u></td>
                <td><b>2.31</b></td>
                <td><b>2.10</b></td>
                <td><b>65.5</b></td>
                <td><u>70.0</u></td>
                <td><b>59.0</b></td>
                <td><b>65.0</b></td>
              </tr>
            </tbody>
          </table>
        </div>
        <div style="font-size:0.85em; color:#555; margin-top:0.5em; text-align:center;">
          <b>Note:</b> Best scores in each category are in <b>bold</b>, second best are <u>underlined</u>.
          <sup>1</sup> from <a href="https://arxiv.org/abs/2503.14182" target="_blank"
            style="color: #1976d2;">BridgeAD</a>, <sup>2</sup> from <a href="https://arxiv.org/abs/2311.15260"
            target="_blank" style="color: #1976d2;">NeuRAD</a><br>
          The improvements in both the overall NeuroNCAP score and, crucially, the reduction in collision rates suggest
          that our dataset helps the model develop a more nuanced understanding of complex road interactions, leading to
          more robust and safer driving policies.
        </div>
      </div>

### 📥 Download Pre-trained Models
  <div style="max-width:900px; margin: 0 auto 2em auto; display: block;">
    <div style="overflow-x: auto;">
      <table class="table is-bordered is-striped is-narrow is-hoverable" style="font-size:0.95em; width:100%;">
        <caption style="caption-side:top; text-align:center; font-weight:bold; margin-bottom:0.5em;">
          Pre-trained Models Download Links
        </caption>
        <thead>
          <tr>
            <th style="width:50%;">Method</th>
            <th style="width:50%;">Download</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>3B Base+nuScenes</td>
            <td><a href="https://huggingface.co/aaaaaap/ImpromptuVLAModel/tree/main/3B_Base_finetune" target="_blank">HF Hub</a></td>
          </tr>
          <tr>
            <td>3B Base+Impromptu</td>
            <td><a href="https://huggingface.co/aaaaaap/ImpromptuVLAModel/tree/main/3B_AD" target="_blank">HF Hub</a></td>
          </tr>
          <tr>
            <td>3B Base+Impromptu+nuScenes</td>
            <td><a href="https://huggingface.co/aaaaaap/ImpromptuVLAModel/tree/main/3B_AD_finetune" target="_blank">HF Hub</a></td>
          </tr>
          <tr>
            <td>7B Base+nuScenes</td>
            <td><a href="https://huggingface.co/aaaaaap/ImpromptuVLAModel/tree/main/7B_Base_finetune" target="_blank">HF Hub</a></td>
          </tr>
          <tr>
            <td>7B Base+Impromptu</td>
            <td><a href="https://huggingface.co/aaaaaap/ImpromptuVLAModel/tree/main/7B_AD" target="_blank">HF Hub</a></td>
          </tr>
          <tr>
            <td>7B Base+Impromptu+nuScenes</td>
            <td><a href="https://huggingface.co/aaaaaap/ImpromptuVLAModel/tree/main/7B_AD_finetune" target="_blank">HF Hub</a></td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

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

### 🎯 Prompts
The prompts we use can be found in [prompts](prompts.md).

### 📊 Close-loop Evaluation with NeuroNCAP

To understand the system's performance within a closed-loop simulation environment, delve into the specifics of our NeuroNCAP-based evaluation: [Close-loop Evaluation](neuroncap_evaluation/evaluation.md) 🎮

### 🎬 Video Gallery
The videos compare the driving behavior of the two models in three representative challenging scenarios: stationary, frontal, and side. For each scenario, **the left column shows the behavior of the base model, which is fine-tuned on nuScenes. The right column shows the performance of the model trained on a subset of our proposed dataset and then fine-tuned on nuScenes**. Compared to the base model, the model using our data can better avoid vehicles by turning, slowing down, etc.

#### stationary

Base+nuScenes&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Base+Impromptu+nuScenes  
<img src="assets/gifs/stationary/1.gif" width="100%" style="max-width: 100%; height: auto;" />
<img src="assets/gifs/stationary/2.gif" width="100%" style="max-width: 100%; height: auto;" />
<img src="assets/gifs/stationary/3.gif" width="100%" style="max-width: 100%; height: auto;" />

#### side

Base+nuScenes&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Base+Impromptu+nuScenes  
<img src="assets/gifs/side/1.gif" width="100%" style="max-width: 100%; height: auto;" />
<img src="assets/gifs/side/2.gif" width="100%" style="max-width: 100%; height: auto;" />
<img src="assets/gifs/side/3.gif" width="100%" style="max-width: 100%; height: auto;" />

#### frontal

Base+nuScenes&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Base+Impromptu+nuScenes  
<img src="assets/gifs/frontal/1.gif" width="100%" style="max-width: 100%; height: auto;" />
<img src="assets/gifs/frontal/2.gif" width="100%" style="max-width: 100%; height: auto;" />
<img src="assets/gifs/frontal/3.gif" width="100%" style="max-width: 100%; height: auto;" />
