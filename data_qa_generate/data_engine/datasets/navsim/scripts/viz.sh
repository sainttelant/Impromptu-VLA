cd /baai-cwm-1/baai_cwm_ml/algorithm/huanang.gao/workspace/202501-drivelm-dpo/DriveEMMA
export PYTHONPATH=$PYTHONPATH:/baai-cwm-1/baai_cwm_ml/algorithm/huanang.gao/workspace/202501-drivelm-dpo/DriveEMMA
export PATH=/baai-cwm-1/baai_cwm_ml/algorithm/huanang.gao/env/envs/driveemma/bin/:$PATH

python data_engine/datasets/navsim/scripts/viz_navsim_all.py \
    --jsonl_file /baai-cwm-1/baai_cwm_ml/cwm/huanang.gao/shared/chenyu.liu/workspace/infer_results/Qwen2___5-VL-7B-Instruct/full/sft/navsim_train_camera_ego_scene_road_metaplanning_planning/checkpoint-900/predict.jsonl \
    --output_file ./log/eval_navsim_result.json \
    --viz_every 100