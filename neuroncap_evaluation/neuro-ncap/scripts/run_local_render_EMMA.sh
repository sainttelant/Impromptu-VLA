#!/bin/bash
seq=${1:?"No sequence specified"}
output_name=${2:?"No output name given (for logging)"}

# loop over all remaining args and check if "--spoof-renderer or --spoof_renderer" is in  ${@:4}
# if it is, set RENDERER_ARGS="--spoof-renderer" and remove it from the list of args
SHOULD_START_RENDERER=true
for arg in ${@:3}; do
  if [[ $arg == "--spoof-renderer" || $arg == "--spoof_renderer" ]]; then
    SHOULD_START_RENDERER=false
  fi
done

# loop over all remaining args and check if "--spoof-renderer or --spoof_renderer" is in  ${@:4}
# if it is, set RENDERER_ARGS="--spoof-renderer" and remove it from the list of args
SHOULD_START_MODEL=true
for arg in ${@:3}; do
  if [[ $arg == "--spoof-model" || $arg == "--spoof_model" ]]; then
    SHOULD_START_MODEL=false
  fi
done

# find two free ports
find_free_port() {
  python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

renderer_port=$(find_free_port)
model_port=$(find_free_port)





if [ $SHOULD_START_RENDERER == true ]; then
  echo "Running NeuRAD service locally..."
  

  python /path/to/your/NeuroNcap/neurad-studio/nerfstudio/scripts/closed_loop/main.py \
    --port $renderer_port \
    --load-config $RENDERING_CHECKPOITNS_PATH/$seq/config.yml \
    --adjust_pose \
    $RENDERER_ARGS \
    &

  RENDERER_PID=$!
  echo "Renderer service started locally with PID $RENDERER_PID"
fi


if [ $SHOULD_START_MODEL == true ]; then

  echo "! Running EMMA-AD service locally..."
  python /path/to/your/NeuroNcap/EMMA-AD/inference/server.py \
    --port $model_port \
    --config_path $MODEL_CFG_PATH \
    --checkpoint_path $MODEL_CHECKPOINT_PATH \
    --qwen_infer_port $qwen_infer_port \
    --past_pos_path $PAST_POS_PATH \
    $MODEL_ARGS \
    &

  MODEL_PID=$!
  echo "Model service started locally with PID $MODEL_PID"
fi



echo "Running neuro-ncap in foreground..."

python /path/to/your/NeuroNcap/neuro-ncap/main.py \
  --engine.renderer.port $renderer_port \
  --engine.model.port $model_port \
  --engine.dataset.data_root $NUSCENES_PATH \
  --engine.dataset.version v1.0-trainval \
  --engine.dataset.sequence $seq \
  --engine.logger.log-dir outoutput/$TIME_NOW/$output_name-$seq \
  ${@:3}




echo "Killing background processes..."
if [ $SHOULD_START_RENDERER == true ]; then
 kill $RENDERER_PID
fi
if [ $SHOULD_START_MODEL == true ]; then
 kill $MODEL_PID
fi