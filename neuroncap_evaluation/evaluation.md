# NeuroNcap Evaluation for Impromptu-VLA

## Installation

```bash
mkdir NeuroNcap
cd NeuroNcap
git clone https://github.com/atonderski/neuro-ncap.git
git clone https://github.com/georghess/neurad-studio.git
```

Then, copy the `EMMA-AD` folder into the `NeuroNcap` directory.
Next, copy the `run_local_render_EMMA.sh` script into the `NeuroNcap/neuro-ncap/scripts` directory.
Also, copy the `run_EMMA.sh` script into the `NeuroNcap/neuro-ncap` directory.

## Preperation

For more details on how to run the evaluation, please refer to the official documentation:  [NeuroNcap official documentation](https://github.com/atonderski/neuro-ncap/blob/main/docs/how-to-run.md)

The above-mentioned document includes details such as environment setup for `neuro-ncap`, checkpoint downloads, and more.

## Environment Setup

In addition to the environment required for running `NeuroNcap`, you also need to set up a separate environment for configuring the inference API of `sglang`. All occurrences of `/path/to/your/sglang_env` in the code should be replaced with the actual path to your `sglang` environment.  
Please refer to the official `sglang` [documentation](https://docs.sglang.ai/start/install.html) for installation instructions.

## Run the evaluation

Before running the evaluation scripts, make sure to replace all `/path/to/your/xx` placeholders in `run_EMMA.sh`, `run_local_render_EMMA.sh`, and the scripts under `EMMA-AD/inference/` with the actual paths on your system.


```bash
cd NeuroNcap/neuro-ncap
bash run_EMMA.sh /path/to/your/qwen_ckpt
```

In `NeuroNcap/neuro-ncap/neuro_ncap/components/evaluator.py`,  the `evaluate_future_collisions` variable in the `EvaluatorConfig` class should be set to `True`  to align with our experimental setup.