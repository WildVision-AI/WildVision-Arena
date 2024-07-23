

## Installation 

```bash 

sudo apt update
sudo apt install cargo
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install bore-cli 
```


```bash
# tmole 10001 & 
# # 4twiln-ip-64-156-70-153.tunnelmole.net 
# tmole 10002 &

# LOCAL_PORT_QWEN=31004
# LOCAL_PORT_COGVLM=31006
# LOCAL_PORT_MINICPM=31007
# LOCAL_PORT_LLAVA7B=31009
# LOCAL_PORT_LLAVA34B=31008
# LOCAL_PORT_BLIP=31012
# LOCAL_PORT_UFORM=31011
# LOCAL_PORT_TINYLLAVA=31015
# LOCAL_PORT_DEEPSEEK=31010
# LOCAL_PORT_BUNNY=31013

# bore local $LOCAL_PORT_QWEN --to 34.19.37.54 & 
# bore local $LOCAL_PORT_COGVLM  --to 34.19.37.54 & 
# bore local $LOCAL_PORT_MINICPM --to 34.19.37.54 & 
# bore local $LOCAL_PORT_LLAVA7B --to 34.19.37.54 & 
# bore local $LOCAL_PORT_LLAVA34B --to 34.19.37.54 & 
# bore local $LOCAL_PORT_BLIP --to 34.19.37.54 & 
# bore local $LOCAL_PORT_UFORM --to 34.19.37.54 & 
# bore local $LOCAL_PORT_TINYLLAVA --to 34.19.37.54 & 
# bore local $LOCAL_PORT_DEEPSEEK --to 34.19.37.54 & 
# bore local $LOCAL_PORT_BUNNY --to 34.19.37.54 & 



# bore local 21004 --to 34.19.37.54 & # 
```

## Deploy 

```bash
bash scripts/run_bore.sh
# bash scripts/restart_models.sh qwen &
bash scripts/restart_models.sh cogvlm & 
bash scripts/restart_models.sh minicpm  &
bash scripts/restart_models.sh llava7b &
bash scripts/restart_models.sh llava34b &
bash scripts/restart_models.sh uform &
bash scripts/restart_models.sh tinyllava  &
bash scripts/restart_models.sh deepseek  &
bash scripts/restart_models.sh bunny  &
```


## Previously

**Qwen-VL-Chat**
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m arena.serve.model_worker --model-path Qwen/Qwen-VL-Chat --controller http://34.19.37.54:8888 --port 31004 --worker http://34.19.37.54:5523 --host=0.0.0.0
```

**CogVLM**

```bash
conda create -n cogvlm python==3.9
conda activate cogvlm
pip install torch==2.1.0 transformers==4.35.0 accelerate==0.24.1 sentencepiece==0.1.99 einops==0.7.0 xformers==0.0.22.post7 triton==2.1.0 uvicorn pillow icecream fastapi protobuf torchvision 
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install attrdict timm==0.9.16

CUDA_VISIBLE_DEVICES=1 python3 -m arena.serve.model_worker --model-path THUDM/cogvlm-chat-hf --controller http://34.19.37.54:8888 --port 31006 --worker http://34.19.37.54:5557 --host=0.0.0.0

# run a restart method 

```

**MiniCPM-V**
```bash
conda create -n minicpm python==3.10
conda activate minicpm
pip install Pillow==10.1.0 timm==0.9.10 torch==2.1.2 torchvision==0.16.2 transformers==4.36.0 sentencepiece==0.1.99 uvicorn pillow icecream fastapi protobuf torchvision psutil accelerate
pip install python-dateutil
pip install attrs
pip install attrdict

CUDA_VISIBLE_DEVICES=0 python3 -m arena.serve.model_worker --model-path openbmb/MiniCPM-V --controller http://34.19.37.54:8888 --port 31007 --worker http://34.19.37.54:5575 --host=0.0.0.0
```


**LLaVAv1.6** (34b and 7b)
some implementation issue here that may require you use 2 gpu for load llava-v1.6-34b for now
```bash
<!-- you might wannner disable flashattn if you encouter error "untimeError: Failed to import transformers.models.llama.modeling_llama because of the following error (look up to see its traceback):                              
.conda/envs/arena/lib/python3.9/site-packages/flash_attn_2_cuda.cpython-39-x86_64-linux-gnu.so: undefined symbol" .conda/envs/arena/lib/python3.9/site-packages/flash_attn/flash_attn_interface.py -->
<!-- easy workaround is by force is_flash_attn_2_available to be False in .conda/envs/arena/lib/python3.9/site-packages/transformers/models/llama/modeling_llama.py -->


CUDA_VISIBLE_DEVICES=3,4 python3 -m arena.serve.model_worker --model-path liuhaotian/llava-v1.6-34b --controller http://34.19.37.54:8888 --port 31008 --worker http://34.19.37.54:5520 --host=127.0.0.1
 
CUDA_VISIBLE_DEVICES=2 python3 -m arena.serve.model_worker --model-path liuhaotian/llava-v1.6-vicuna-7b --controller http://34.19.37.54:8888 --port 31009 --worker http://34.19.37.54:5591 --host=127.0.0.1
```

**BLIP**
```bash
CUDA_VISIBLE_DEVICES=2 python3 -m arena.serve.model_worker --model-path Salesforce/instructblip-vicuna-7b --controller http://34.19.37.54:8888 --port 31012 --worker http://34.19.37.54:5535 --host=0.0.0.0
```

**UForm**
```bash
conda create -n uform python==3.10
conda activate uform
pip install uform uvicorn datasets psutil accelerate icecream fastapi
CUDA_VISIBLE_DEVICES=5 python3 -m arena.serve.model_worker --model-path unum-cloud/uform-gen2-qwen-500m --controller http://34.19.37.54:8888 --port 31011 --worker http://34.19.37.54:5596 --host=0.0.0.0
```

**TinyLLaVA**
```bash
# Make sure to have transformers >= 4.35.3.
CUDA_VISIBLE_DEVICES=0 python3 -m arena.serve.model_worker --model-path bczhou/tiny-llava-v1-hf --controller http://34.19.37.54:8888 --port 31015 --worker http://34.19.37.54:5525 --host=0.0.0.0 --num-gpus 1
```

**DeepSeek VL**
```bash
conda env create -f model_config/deepseek_vl.yml
conda activate deepseek_vl
pip install -e .
pip install datasets  icecream
CUDA_VISIBLE_DEVICES=5 python3 -m arena.serve.model_worker --model-path deepseek-ai/deepseek-vl-7b-chat --controller http://34.19.37.54:8888 --port 31010 --worker http://34.19.37.54:5555 --host=0.0.0.0 --num-gpus 1
```

**Bunny**
```bash
CUDA_VISIBLE_DEVICES=5 python3 -m arena.serve.model_worker --model-path BAAI/Bunny-v1_0-3B --controller http://34.19.37.54:8888 --port 31013 --worker http://34.19.37.54:5543 --host=0.0.0.0 --num-gpus 1
```




--- 

**Yi-VL** (Work?)
```bash
conda env create -f model_config/yi_vl.yml
conda activate yi_vl
pip install -e .
CUDA_VISIBLE_DEVICES=5 python3 -m arena.serve.model_worker --model-path 01-ai/Yi-VL-6B --controller http://34.19.37.54:8888 --port 31014 --worker http://34.19.37.54:31014 --host=0.0.0.0 --num-gpus 1
https://huggingface.co/01-ai/Yi-VL-6B/discussions/7
```

