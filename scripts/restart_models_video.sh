#!/bin/bash
model_name=$1

GCP_IP=34.19.37.54

LOCAL_PORT_QWEN=31004
LOCAL_PORT_COGVLM=31006
LOCAL_PORT_MINICPM=31007
LOCAL_PORT_LLAVA7B=31009
LOCAL_PORT_LLAVA34B=31008
LOCAL_PORT_BLIP=31012
LOCAL_PORT_UFORM=31011
LOCAL_PORT_TINYLLAVA=31015
LOCAL_PORT_DEEPSEEK=31010
LOCAL_PORT_BUNNY=31013

# copied from `run_bore.sh` results
# PORT_QWEN=5584
# PORT_BUNNY=5586
# PORT_TINYLLAVA=5525
# PORT_UFORM=5563
# PORT_LLAVA7B=5590
# PORT_COGVLM=5510
# PORT_DEEPSEEK=5507
# PORT_MINICPM=5523
# PORT_BLIP=5537
# PORT_LLAVA34B=5558

PORT_BUNNY=5571
PORT_TINYLLAVA=5547
PORT_UFORM=5530
PORT_LLAVA7B=5518
PORT_COGVLM=5572
PORT_DEEPSEEK=5560
PORT_MINICPM=5525
PORT_BLIP=5565
PORT_LLAVA34B=5573


# Infinite loop to restart the command every 10 minutes
while true; do
    # Start the Python command in the background
    SLEEP_TIME=10800
    if [ "$model_name" = "cogvlm" ]; then
        pkill -f $PORT_COGVLM
        CUDA_VISIBLE_DEVICES=1 /home/yuchenl/.conda/envs/cogvlm/bin/python -m arena.serve.model_worker --model-path THUDM/cogvlm-chat-hf --controller http://${GCP_IP}:8888 --port $LOCAL_PORT_COGVLM --worker http://${GCP_IP}:${PORT_COGVLM} --host=0.0.0.0 &
    elif [ "$model_name" = "qwen" ]; then
        pkill -f $PORT_QWEN
        CUDA_VISIBLE_DEVICES=0 /home/yuchenl/.conda/envs/wildvision-arena/bin/python -m arena.serve.model_worker --model-path Qwen/Qwen-VL-Chat --controller http://${GCP_IP}:8888 --port $LOCAL_PORT_QWEN --worker http://${GCP_IP}:${PORT_QWEN}  --host=0.0.0.0 &
    elif [ "$model_name" = "minicpm" ]; then
        pkill -f $PORT_MINICPM
        CUDA_VISIBLE_DEVICES=0 /home/yuchenl/.conda/envs/minicpm/bin/python -m arena.serve.model_worker --model-path openbmb/MiniCPM-V --controller http://${GCP_IP}:8888 --port $LOCAL_PORT_MINICPM --worker http://${GCP_IP}:${PORT_MINICPM} --host=0.0.0.0 &
    elif [ "$model_name" = "llava7b" ]; then
        pkill -f $PORT_LLAVA7B
        CUDA_VISIBLE_DEVICES=2 /home/yuchenl/.conda/envs/wildvision-arena/bin/python -m arena.serve.model_worker --model-path liuhaotian/llava-v1.6-vicuna-7b --controller http://${GCP_IP}:8888 --port $LOCAL_PORT_LLAVA7B --worker http://${GCP_IP}:${PORT_LLAVA7B} --host=0.0.0.0 &
    elif [ "$model_name" = "llava34b" ]; then
        pkill -f $PORT_LLAVA34B
        CUDA_VISIBLE_DEVICES=3,4 /home/yuchenl/.conda/envs/wildvision-arena/bin/python -m arena.serve.model_worker --model-path liuhaotian/llava-v1.6-34b --controller http://${GCP_IP}:8888 --port $LOCAL_PORT_LLAVA34B --worker http://${GCP_IP}:${PORT_LLAVA34B} --host=0.0.0.0 &
    elif [ "$model_name" = "blip" ]; then
        pkill -f $PORT_BLIP
        CUDA_VISIBLE_DEVICES=2 /home/yuchenl/.conda/envs/wildvision-arena/bin/python -m arena.serve.model_worker --model-path Salesforce/instructblip-vicuna-7b --controller http://${GCP_IP}:8888 --port $LOCAL_PORT_BLIP --worker http://${GCP_IP}:${PORT_BLIP} --host=0.0.0.0 &
    elif [ "$model_name" = "uform" ]; then
        pkill -f $PORT_UFORM
        CUDA_VISIBLE_DEVICES=5 /home/yuchenl/.conda/envs/uform/bin/python -m arena.serve.model_worker --model-path unum-cloud/uform-gen2-qwen-500m --controller http://${GCP_IP}:8888 --port $LOCAL_PORT_UFORM --worker http://${GCP_IP}:${PORT_UFORM} --host=0.0.0.0 &
    elif [ "$model_name" = "tinyllava" ]; then
        pkill -f $PORT_TINYLLAVA
        CUDA_VISIBLE_DEVICES=0 /home/yuchenl/.conda/envs/wildvision-arena/bin/python -m arena.serve.model_worker --model-path bczhou/tiny-llava-v1-hf --controller http://${GCP_IP}:8888 --port $LOCAL_PORT_TINYLLAVA --worker http://${GCP_IP}:${PORT_TINYLLAVA} --host=0.0.0.0 --num-gpus 1 &
    elif [ "$model_name" = "deepseek" ]; then
        pkill -f $PORT_DEEPSEEK
        CUDA_VISIBLE_DEVICES=5 /home/yuchenl/.conda/envs/deepseek_vl/bin/python -m arena.serve.model_worker --model-path deepseek-ai/deepseek-vl-7b-chat --controller http://${GCP_IP}:8888 --port $LOCAL_PORT_DEEPSEEK --worker http://${GCP_IP}:${PORT_DEEPSEEK} --host=0.0.0.0 --num-gpus 1 &
    elif [ "$model_name" = "bunny" ]; then
        pkill -f $PORT_BUNNY
        CUDA_VISIBLE_DEVICES=5 /home/yuchenl/.conda/envs/wildvision-arena/bin/python -m arena.serve.model_worker --model-path BAAI/Bunny-v1_0-3B --controller http://${GCP_IP}:8888 --port $LOCAL_PORT_BUNNY --worker http://${GCP_IP}:${PORT_BUNNY} --host=0.0.0.0 --num-gpus 1 &
    else 
        echo "Invalid model name"
        exit 1
    fi
    # Get the PID of the last background process
    PID=$!

    echo "PID: $PID"
    
    # Wait for 1 hour
    sleep $SLEEP_TIME
    
    echo "Killing PID: $PID"
    # Kill the Python command
    kill $PID
    
    # Wait a bit before restarting (optional, for stability)
    sleep 30 
    echo "Restarting..."
done

# bash scripts/restart_models.sh qwen &
# bash scripts/restart_models.sh cogvlm & 
# bash scripts/restart_models.sh minicpm  &
# bash scripts/restart_models.sh llava7b &
# bash scripts/restart_models.sh llava34b &
# bash scripts/restart_models.sh uform &
# bash scripts/restart_models.sh tinyllava  &
# bash scripts/restart_models.sh deepseek  &
# bash scripts/restart_models.sh bunny  &
