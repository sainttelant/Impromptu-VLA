#################################################################
# Edit the following paths to match your setup
qwen_ckpt_path=$1
BASE_DIR='/path/to/your/NeuroNcap'
NUSCENES_PATH='/path/to/your/NeuroNcap/neuro-ncap/data/nuscenes'
# Model related stuff
MODEL_NAME='EMMA-AD'
MODEL_FOLDER=$BASE_DIR/$MODEL_NAME

# Rendering related stuff
RENDERING_FOLDER=$BASE_DIR/'neurad-studio'
RENDERING_CHECKPOITNS_PATH='checkpoints'

# NCAP related stuff
NCAP_FOLDER=$BASE_DIR/'neuro-ncap'

# EMMA-AD server port file
EMMA_AD_PORT_FILE='/path/to/your/port.txt'

# Evaluation default values, set to lower for debugging original=50
RUNS=1

#################################################################

# SLURM related stuff
TIME_NOW=$(date +"%Y-%m-%d_%H-%M-%S")


# assert we are standing in the right folder, which is NCAP folder
if [ $PWD != $NCAP_FOLDER ]; then
    echo "Please run this script from the NCAP folder"
    exit 1
fi

# assert all the other folders are present
if [ ! -d $MODEL_FOLDER ]; then
    echo "Model folder not found"
    exit 1
fi

if [ ! -d $RENDERING_FOLDER ]; then
    echo "Rendering folder not found"
    exit 1
fi



/path/to/your/sglang_env/bin/python /path/to/your/NeuroNcap/EMMA-AD/inference/launch_server.py --qwen_ckpt_path $qwen_ckpt_path --port_file $EMMA_AD_PORT_FILE

qwen_infer_port=$(cat $EMMA_AD_PORT_FILE)
echo "--------------qwen_infer_port is $qwen_infer_port---------------------"

PAST_POS_PATH="/path/to/your/past_pos.npy"





for SCENARIO in "stationary" "frontal" "side"; do
    array_file=ncap_slurm_array_$SCENARIO
    id_to_seq=scripts/arrays/${array_file}.txt
#stationary_num = 10; frontal = 5; side_num = 5
    if [ $SCENARIO == "stationary" ]; then
        num_scenarios=10
    elif [ $SCENARIO == "frontal" ]; then
        num_scenarios=5
    elif [ $SCENARIO == "side" ]; then
        num_scenarios=5
    fi
    for i in $(seq 1 $num_scenarios); do
        sequence=$(awk -v ArrayTaskID=$i '$1==ArrayTaskID {print $2}' $id_to_seq)
        if [ -z $sequence ]; then
            echo "undefined sequence"
            exit 0
        fi
        echo "Running scenario $SCENARIO with sequence $sequence"
        BASE_DIR=$BASE_DIR\
         NUSCENES_PATH=$NUSCENES_PATH\
         MODEL_NAME=$MODEL_NAME\
         MODEL_FOLDER=$MODEL_FOLDER\
         RENDERING_FOLDER=$RENDERING_FOLDER\
         RENDERING_CHECKPOITNS_PATH=$RENDERING_CHECKPOITNS_PATH\
         NCAP_FOLDER=$NCAP_FOLDER\
         TIME_NOW=$TIME_NOW\
         qwen_infer_port=$qwen_infer_port\
         PAST_POS_PATH=$PAST_POS_PATH\
         scripts/run_local_render_EMMA.sh $sequence $SCENARIO --scenario-category=$SCENARIO --runs $RUNS
        #exit 0
    done
done
kill $MODEL_PID