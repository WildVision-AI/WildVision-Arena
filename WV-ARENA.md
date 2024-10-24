# WildVision-Arena (WV-Arena)
## Custom API
1. Add API_STREAM_ITER (notice that even though the function name are all stream_iter, model stream generation is optional)
```python
# `arena/serve/api_provider.py`, refer to `qwenvl_api_stream_iter`
def your_api_stream_iter(model_name, conv, temperature, top_p, max_new_tokens, image):
    messages = conv.to_yourapi_messages()
    response = model(messages)
    data = {
        "text": response.output.choices[0].message.content[0]["text"],
        "error_code": 0,
    }
    yield data
```

2. Add your API conv template (you can also apply exisiting one if it's the same).
```python
# `arena/conversation.py` refer to `to_qwenvlapi_messages`
    def to_yourapi_messages(self):
        """Convert the conversation to yourapi completion format."""
        if self.system_message == "":
            ret = []
        else:
            ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": [{"text": msg}, {"image": self.media_url}]})
        return ret
```
And then remember to register your model accordingly.
```python
# `arena/conversation.py` refer to `QwenVLAPI template`
register_conv_template(
    Conversation(
        name="your_model_name",
        system_message="",
        roles=("", ""),
        sep_style=SeparatorStyle.RAW,
    )
)
```

3. Add your api stream to bot response.
```python
# `arena/serve/gradio_web_server.py` refer to qwenvl_api_stream_iter
from arena.serve.api_provider import your_api_stream_iter
def bot_response(
    state,
    chatbot,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
    apply_rate_limit=True,
):
    ...
    elif model_name in [
        "your_model_name"
    ]:
        stream_iter = your_api_stream_iter(
            model_name, conv, temperature, top_p, max_new_tokens, image
        )
    ...
```

4. Add your model adapter
```python
# `arena/model/model_adapter.py`, refer to QwenVLAPIAdapter
class YourAPIAdapter(BaseModelAdapter):
    """The model adapter for YourAPI"""

    def match(self, model_path: str):
        return "your_model_name" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        return None, None
    
    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("your_model_name")

register_model_adapter(YourAPIAdapter)
```

5. Test your model via Gradio Demo.

## Gradio Demo

Guideline:
<!-- - We're now comparing `gpt-4-vision-preview` and `llava-v1.5-13b` -->
- Launch a gradio demo for data collection, follow [FastChat](https://github.com/lm-sys/FastChat) for more details.
- Select tab `Arena (side-by-side)` select the two MLLMs you want combat
- Upload image and add text, and click send

### Start the Controller
```bash
export WILDVISION_ARENA_LOGDIR="YOUR_LOG_DIR"
# export DOWNLOAD_DATASET=NA
# export WILDVISION_ARENA_LOGDIR="/home/yuchenl/vision-arena-logs"
python -m arena.serve.controller --host='127.0.0.1' --port 21002 &

```

### Launch the Image-Text Model Worker
**GPT4V, GeminiPro, Claude, Yi-VL-PLUS, Reka**
```bash
export GOOGLE_API_KEY=YOUR_API_KEY
export OPENAI_API_KEY=YOUR_API_KEY
export ANTHROPIC_API_KEY=YOUR_API_KEY

export WEB_IMG_FOLDER="/home/yuchenl/http_img/"
export YIVL_API_KEY=YOUR_API_KEY
export WEB_IMG_URL_ROOT="http://34.19.37.54:5006"
export YIVL_API_BASE="https://api.lingyiwanwu.com/v1"
export LLAVA_API_KEY="EMPTY"
export LLAVA_API_BASE="https://llava-next-endpoint.lmms-lab.com/v1"
pip install reka-api
export REKA_API_KEY=YOUR_API_KEY

export HF_API_TOKEN=YOUR_API_KEY
```

```bash
python3 -m arena.serve.model_worker --model-path gpt-4-vision-preview --controller http://127.0.0.1:21002 --port 31001 --worker http://127.0.0.1:31001 --host=127.0.0.1 &
python3 -m arena.serve.model_worker --model-path gemini-pro-vision --controller http://127.0.0.1:21002 --port 31003 --worker http://127.0.0.1:31003 --host=127.0.0.1 &
python3 -m arena.serve.model_worker --model-path claude-3-opus-20240229 --controller http://127.0.0.1:21002 --port 31016 --worker http://127.0.0.1:31016 --host=127.0.0.1 &
python3 -m arena.serve.model_worker --model-path claude-3-sonnet-20240229 --controller http://127.0.0.1:21002 --port 31019 --worker http://127.0.0.1:31019 --host=127.0.0.1 &
python3 -m arena.serve.model_worker --model-path claude-3-haiku-20240307 --controller http://127.0.0.1:21002 --port 31020 --worker http://127.0.0.1:31020 --host=127.0.0.1 &
python3 -m arena.serve.model_worker --model-path yi-vl-plus --controller http://127.0.0.1:21002 --port 31021 --worker http://127.0.0.1:31021 --host=127.0.0.1 &
# python3 -m arena.serve.model_worker --model-path reka --controller http://127.0.0.1:21002 --port 31022 --worker http://127.0.0.1:31022 --host=127.0.0.1
python3 -m arena.serve.model_worker --model-path Reka-Flash --controller http://127.0.0.1:21002 --port 31022 --worker http://127.0.0.1:31022 --host=127.0.0.1 

# CUDA_VISIBLE_DEVICES=2 python3 -m arena.serve.model_worker --model-path liuhaotian/llava-v1.5-13b --controller http://127.0.0.1:21002 --port 31002 --worker http://127.0.0.1:31002 --host=127.0.0.1  --num-gpus 1 &

python3 -m arena.serve.model_worker --model-path idefics2-8b-chatty --controller http://127.0.0.1:21002 --port 31023 --worker http://127.0.0.1:31023 --host=127.0.0.1

python3 -m arena.serve.model_worker --model-path gpt-4o --controller http://127.0.0.1:21002 --port 31024 --worker http://127.0.0.1:31024 --host=127.0.0.1 &

python3 -m arena.serve.model_worker --model-path gemini-1.5-flash-latest --controller http://127.0.0.1:21002 --port 31025 --worker http://127.0.0.1:31025 --host=127.0.0.1 &

python3 -m arena.serve.model_worker --model-path minicpm-llama3-v --controller http://127.0.0.1:21002 --port 31026 --worker http://127.0.0.1:31026 --host=127.0.0.1 &

python3 -m arena.serve.model_worker --model-path Reka-Core --controller http://127.0.0.1:21002 --port 31027 --worker http://127.0.0.1:31027 --host=127.0.0.1 &

python3 -m arena.serve.model_worker --model-path qwen-vl-max --controller http://127.0.0.1:21002 --port 31028 --worker http://127.0.0.1:31028 --host=127.0.0.1 &
python3 -m arena.serve.model_worker --model-path qwen-vl-plus --controller http://127.0.0.1:21002 --port 31029 --worker http://127.0.0.1:31029 --host=127.0.0.1 &


python3 -m arena.serve.model_worker --model-path gpt-4-turbo --controller http://127.0.0.1:21002 --port 31030 --worker http://127.0.0.1:31030 --host=127.0.0.1 &

python3 -m arena.serve.model_worker --model-path llava-next-72b --controller http://127.0.0.1:21002 --port 31031 --worker http://127.0.0.1:31031 --host=127.0.0.1


```

**Qwen-VL-Chat**
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m arena.serve.model_worker --model-path Qwen/Qwen-VL-Chat --controller http://127.0.0.1:21002 --port 31004 --worker http://127.0.0.1:31004 --host=127.0.0.1 &
```

**CogVLM**

```bash
conda create -n cogvlm python==3.9
conda activate cogvlm
pip install torch==2.1.0 transformers==4.35.0 accelerate==0.24.1 sentencepiece==0.1.99 einops==0.7.0 xformers==0.0.22.post7 triton==2.1.0 uvicorn pillow icecream fastapi protobuf torchvision 
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install attrdict timm==0.9.16

CUDA_VISIBLE_DEVICES=1 python3 -m arena.serve.model_worker --model-path THUDM/cogvlm-chat-hf --controller http://127.0.0.1:21002 --port 31006 --worker http://127.0.0.1:31006 --host=127.0.0.1 

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

CUDA_VISIBLE_DEVICES=0 python3 -m arena.serve.model_worker --model-path openbmb/MiniCPM-V --controller http://127.0.0.1:21002 --port 31007 --worker http://127.0.0.1:31007 --host=127.0.0.1 
```

**InternVL**
```bash
pip install peft flash_attn
CUDA_VISIBLE_DEVICES=0 python3 -m arena.serve.model_worker --model-path OpenGVLab/InternVL-Chat-V1-5 --controller http://127.0.0.1:21002 --port 31026 --worker http://127.0.0.1:31026 --host=127.0.0.1 
```

**LLaVAv1.6**
some implementation issue here that may require you use 2 gpu for load llava-v1.6-34b for now
```bash
<!-- If you encouter error "untimeError: Failed to import transformers.models.llama.modeling_llama because of the following error (look up to see its traceback):                              
.conda/envs/arena/lib/python3.9/site-packages/flash_attn_2_cuda.cpython-39-x86_64-linux-gnu.so: undefined symbol" .conda/envs/arena/lib/python3.9/site-packages/flash_attn/flash_attn_interface.py, you might wannner 1. try older version of flash-attn or 2. simply disable flashattn  -->
CUDA_VISIBLE_DEVICES=3,4 python3 -m arena.serve.model_worker --model-path liuhaotian/llava-v1.6-34b --controller http://127.0.0.1:21002 --port 31008 --worker http://127.0.0.1:31008 --host=127.0.0.1

# CUDA_VISIBLE_DEVICES=5 python3 -m arena.serve.model_worker --model-path liuhaotian/llava-v1.6-vicuna-13b --controller http://127.0.0.1:21002 --port 31010 --worker http://127.0.0.1:31010 --host=127.0.0.1

CUDA_VISIBLE_DEVICES=2 python3 -m arena.serve.model_worker --model-path liuhaotian/llava-v1.6-vicuna-7b --controller http://127.0.0.1:21002 --port 31011 --worker http://127.0.0.1:31011 --host=127.0.0.1
```

**BLIP**
```bash
CUDA_VISIBLE_DEVICES=2 python3 -m arena.serve.model_worker --model-path Salesforce/instructblip-vicuna-7b --controller http://127.0.0.1:21002 --port 31012 --worker http://127.0.0.1:31012 --host=127.0.0.1 
```

**UForm**
```bash
conda create -n uform python==3.10
conda activate uform
pip install uform uvicorn datasets psutil accelerate icecream fastapi
CUDA_VISIBLE_DEVICES=2 python3 -m arena.serve.model_worker --model-path unum-cloud/uform-gen2-qwen-500m --controller http://127.0.0.1:21002 --port 31013 --worker http://127.0.0.1:31013 --host=127.0.0.1 
```

**Yi-VL**
```bash
git clone https://github.com/01-ai/Yi.git
cd Yi ; mkdir model ; cd model
git lfs clone https://huggingface.co/01-ai/Yi-VL-6B
cd Yi/VL
pip install -r requirements.txt
cd ~/WildVision-Arena
pip install -e .
# CUDA_VISIBLE_DEVICES=3 python debug_model/test_yivl.py --model-path /local/home/yujielu/project/Yi/model/Yi-VL-6B --image-file image.jpg --question "Describe the image in detail."
CUDA_VISIBLE_DEVICES=3 python3 -m arena.serve.model_worker --model-path /local/home/yujielu/project/Yi/model/Yi-VL-6B --controller http://127.0.0.1:21002 --port 31014 --worker http://127.0.0.1:31014 --host=127.0.0.1  --num-gpus 1
```

**TinyLLaVA**
```bash
# Make sure to have transformers >= 4.35.3.
CUDA_VISIBLE_DEVICES=0 python3 -m arena.serve.model_worker --model-path bczhou/tiny-llava-v1-hf --controller http://127.0.0.1:21002 --port 31015 --worker http://127.0.0.1:31015 --host=127.0.0.1  --num-gpus 1
```

**DeepSeek VL**
```bash
conda env create -f model_config/deepseek_vl.yml
conda activate deepseek_vl
pip install -e .
pip install datasets  icecream
CUDA_VISIBLE_DEVICES=5 python3 -m arena.serve.model_worker --model-path deepseek-ai/deepseek-vl-7b-chat --controller http://127.0.0.1:21002 --port 31017 --worker http://127.0.0.1:31017 --host=127.0.0.1  --num-gpus 1
```

**Bunny**
```bash
CUDA_VISIBLE_DEVICES=3 python3 -m arena.serve.model_worker --model-path BAAI/Bunny-v1_0-3B --controller http://127.0.0.1:21002 --port 31018 --worker http://127.0.0.1:31018 --host=127.0.0.1  --num-gpus 1
```

**Idefics2**
'idefics2-8b-chatty'
Notices:
1. As of April 18th, 2024, Idefics2 is part of the 4.40.0 Transformers pypi release. Please upgrade your Transformers version (pip install transformers --upgrade).
2. You many need to try other version of flash-attn when encountering error
```bash
pip install transformers --upgrade
pip install flash-attn --no-build-isolation

python3 -m arena.serve.model_worker --model-path idefics2 --controller http://127.0.0.1:21002 --port 31023 --worker http://127.0.0.1:31023 --host=127.0.0.1

GCP_IP=34.19.37.54 
LOCAL_PORT_IDEFICS2=31013 # must be a public port such that the GCP_IP can access it
CUDA_VISIBLE_DEVICES=0 python -m arena.serve.model_worker --model-path idefics2 --controller http://${GCP_IP}:8888 --port $LOCAL_PORT_IDEFICS2 --worker http://127.0.0.1:${LOCAL_PORT_IDEFICS2} --host=0.0.0.0 &
```

### Launch the Video Text Model Worker

**[Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)**
Take model [LanguageBind/Video-LLaVA-7B](https://huggingface.co/LanguageBind/Video-LLaVA-7B) as an example.
```bash
conda env create -f model_config/videollava.yml
conda activate arena-videollava
# Test your enviroment by run inference: python inference/test_videollava.py
# Notice that for VideoLLM, port number start from 32001
CUDA_VISIBLE_DEVICES=0 python3 -m arena.serve.model_worker --model-path LanguageBind/Video-LLaVA-7B --controller http://127.0.0.1:21002 --port 32001 --worker http://127.0.0.1:32001 --host=127.0.0.1  --num-gpus 1
```


**[VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2)**
Take model [DAMO-NLP-SG/VideoLLaMA2-7B](https://drive.google.com/drive/u/0/folders/1-P7p-tq5aXZzSoefEJx4PSFKH8jt8KWy) as an example.
```bash
conda env create -f model_config/videollama2.yml
conda activate arena-videollama2
CUDA_VISIBLE_DEVICES=0 python3 -m arena.serve.model_worker --model-path DAMO-NLP-SG/VideoLLaMA2-7B --controller http://127.0.0.1:21002 --port 32002 --worker http://127.0.0.1:32002 --host=127.0.0.1  --num-gpus 1
```

**[LITA](https://github.com/NVlabs/LITA)**
Take model [LITA-13B-v1.3](https://huggingface.co/LanguageBind/Video-LLaVA-7B) as an example.
```bash
conda env create -f model_config/lita.yml
conda activate arena-lita
# Test your enviroment by run inference: python inference/test_lita.py --model-path /private/home/yujielu/downloads/weights/lita-vicuna-v1-3-13b-finetune --visual-path /private/home/yujielu/project/WildVision-Arena/examples/dancing.mp4 --visual-data-type video
CUDA_VISIBLE_DEVICES=0 python3 -m arena.serve.model_worker --model-path LITA-13B-v1.3 --controller http://127.0.0.1:21002 --port 32003 --worker http://127.0.0.1:32003 --host=127.0.0.1  --num-gpus 1
```

**[LLaVANeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)**
Take model [LLaVA-NeXT-Video-7B](https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-7B) as an example.
```
CUDA_VISIBLE_DEVICES=0 python3 -m arena.serve.model_worker --model-path lmms-lab/LLaVA-NeXT-Video-7B --controller http://127.0.0.1:21002 --port 32004 --worker http://127.0.0.1:32004 --host=127.0.0.1  --num-gpus 1
```

**[VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2)**
Take model [VideoLLaMA2-7B](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2-7B) as an example.
```
CUDA_VISIBLE_DEVICES=0 python3 -m arena.serve.model_worker --model-path DAMO-NLP-SG/VideoLLaMA2-7B --controller http://127.0.0.1:21002 --port 32005 --worker http://127.0.0.1:32005 --host=127.0.0.1  --num-gpus 1
```

### Start the Web Server

```bash
python -m arena.serve.gradio_web_server_multi --share --port 8688 --controller-url http://127.0.0.1:21002 --show-terms-of-use
python -m arena.serve.gradio_web_server_multi --share --controller-url http://127.0.0.1:21002 --elo-results-file ./elo_results.pkl --leaderboard-table-file ./leaderboard.csv &


python -m arena.serve.gradio_web_server_multi --share --port 1213 --controller-url http://127.0.0.1:21002 

# python -m arena.serve.gradio_web_server_multi --share --controller-url http://127.0.0.1:21002 --elo-results-file /home/yuchenl/Arena-Elo/results/latest/elo_results.pkl --leaderboard-table-file /leaderboard.csv &
```



## Leaderboard

### Monitoring 

```bash

```

### Testing Our Evaluator
```bash
CUDA_VISIBLE_DEVICES=4 python3 -m arena.serve.model_worker --model-path /share/edc/home/yujielu/project/uniscore_data/output_checkpoints/llava/llava-v1.5_evaluator --controller http://127.0.0.1:21002 --port 31020 --worker http://127.0.0.1:31020 --host=127.0.0.1  --num-gpus 1
```


### Precompute generations

```bash
# launch by model name
CUDA_VISIBLE_DEVICES=0 python -m arena.balance_elo_rating.precompute --model_name "liuhaotian/llava-v1.6-vicuna-7b"
# launch by worker addr
python -m arena.balance_elo_rating.precompute --worker_addr "http://{worker_addr}:{port}" 
```
Generation results are saved in `./arena/balance_elo_rating/gen_results/{model_name}`

To load the precomputed generations, use the following command:
```python
import datasets
dataset = datasets.load_from_disk('./arena/balance_elo_rating/gen_results/{model_name}')
"""
Dataset({
    features: ['question_id', 'model', 'conversation', 'language', 'image', 'turn'],
    num_rows: 5
})
"""
```

### Evaluator
**Balanced Elo Rating**
```bash
# Submit battle release subset as a test leaderboard
python scripts/upload_woprecompute_hf.py
# Submit battle release subset as a test leaderboard, with balanced elo rating, using gpt4v on precomputed conversations
python scripts/upload_precompute_hf.py
```
**Local Evaluator**
```bash
CUDA_VISIBLE_DEVICES=2 python -m arena.balance_elo_rating.local_evaluate --model_name "/share/edc/home/yujielu/project/uniscore_data/output_checkpoints/llava-v1.5/llava_evaluator_v2_0222"
```


### Model Eval
**Evaluate Model**
```bash
CUDA_VISIBLE_DEVICES=2 python -m arena.balance_elo_rating.precompute --model_name "/share/edc/home/yujielu/ckpt_hub/otask_exp/exp/output_checkpoints/llava-v1.5/llava_625k_woimg0.95_1e5_1eps_bsz80/llava-v1.5-7b-095"
```
