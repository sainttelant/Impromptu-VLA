# NeuroNCAP Evaluation for Impromptu-VLA

## Installation

```bash
mkdir NeuroNCAP
cd NeuroNCAP
git clone https://github.com/atonderski/neuro-ncap.git
git clone https://github.com/georghess/neurad-studio.git
```

Then, copy the `Impromptu` folder into the `NeuroNCAP` directory.
Next, copy the `run_local_render.sh` script into the `NeuroNCAP/neuro-ncap/scripts` directory.
Also, copy the `run.sh` script into the `NeuroNCAP/neuro-ncap` directory.

## Preperation

For more details on how to run the evaluation, please refer to the official documentation:  [NeuroNCAP official documentation](https://github.com/atonderski/neuro-ncap/blob/main/docs/how-to-run.md)

The above-mentioned document includes details such as environment setup for `neuro-ncap`, checkpoint downloads, and more.

## Environment Setup

In addition to the environment required for running `NeuroNCAP`, you also need to set up a separate environment for configuring the inference API of `sglang`. All occurrences of `/path/to/your/envs/sglang` in the code should be replaced with the actual path to your `sglang` environment.  
Please refer to the official `sglang` [documentation](https://docs.sglang.ai/start/install.html) for installation instructions.

The original repository uses Docker. We provide a method to directly build a virtual environment without Docker. We provide the .yaml files for neuro-ncap and neurad-studio to facilitate environment construction. They are [neuro-ncap.yaml](neuroncap_evaluation/neuro-ncap.yaml) and [neurad-studio.yaml](neuroncap_evaluation/neurad-studio.yaml) respectively.

In addition, the previous code could not specify RUNS, NAME, file_path, etc. through parameters. We have now updated the code to support passing these variables through parameters. For specific usage, please refer to [run.sh](neuroncap_evaluation/run.sh)

## Run the evaluation

Before running the evaluation scripts, make sure to replace all `/path/to/your/xx` placeholders in `run.sh`, `run_local_render.sh`, and the scripts under `Impromptu/inference/` with the actual paths on your system.


```bash
cd NeuroNCAP/neuro-ncap
bash run.sh /path/to/your/qwen_ckpt RUNS CUSTOM_SUFFIX
```

In `NeuroNCAP/neuro-ncap/neuro_ncap/components/evaluator.py`,  the `evaluate_future_collisions` variable in the `EvaluatorConfig` class should be set to `True`  to align with our experimental setup.