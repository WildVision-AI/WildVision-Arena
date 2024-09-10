## Installation 


```bash
#API keys

export GOOGLE_API_KEY=YOUR_API_KEY
export OPENAI_API_KEY=YOUR_API_KEY
export ANTHROPIC_API_KEY=YOUR_API_KEY

export WEB_IMG_FOLDER="/home/yuchenl/http_img/"
export YIVL_API_KEY=YOUR_API_KEY
export WEB_IMG_URL_ROOT="http://34.19.37.54:5006"
export YIVL_API_BASE="https://api.lingyiwanwu.com/v1"

pip install reka-api
export REKA_API_KEY=YOUR_API_KEY

export HF_API_TOKEN=YOUR_API_KEY
# 
```

```bash
# sudo mount /dev/sdb /mnt/disks/logs/

sudo apt update
sudo apt install cargo
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install bore-cli
bore server --min-port  5500 --max-port 5600
```

```bash
conda create -n wildvision-arena python=3.9
conda activate wildvision-arena
pip install -r requirements.txt
pip install -q -U google-generativeai
pip install transformers-stream-generator
```

## Controller

```bash  
export WILDVISION_ARENA_LOGDIR="/home/yuchenl/vision-arena-logs"
python -m arena.serve.controller --host='0.0.0.0' --port 8888 
```


## Gradio 
<!-- python -m arena.serve.gradio_web_server_multi --port 5000 --controller-url http://0.0.0.0:8888 --elo-results-file ./elo_results.pkl --leaderboard-table-file ./leaderboard.csv --share -->

<!-- python -m arena.serve.gradio_web_server_multi --share --controller-url http://34.19.37.54:8888 --elo-results-file ./elo_results.pkl --leaderboard-table-file ./leaderboard.csv -->

<!-- python -m arena.serve.gradio_web_server_multi --share --controller-url http://0.0.0.0:8888 --elo-results-file ./elo_results.pkl --leaderboard-table-file ./leaderboard.csv & -->

python -m arena.serve.gradio_web_server_multi_new --port 5679 --controller-url http://0.0.0.0:8888 --elo-results-file ./elo_results.pkl --leaderboard-table-file ./leaderboard.csv  

<!-- http://34.19.37.54:5679/  -->

bash scripts/restart_gradio.sh

## Check online

http://34.19.37.54:5000

## API Models on GCP

```bash
# python3 -m arena.serve.model_worker --model-path gemini-pro-vision --controller http://127.0.0.1:8888 --port 31003 --worker http://127.0.0.1:31003 --host=127.0.0.1 &
python3 -m arena.serve.model_worker --model-path gpt-4-vision-preview --controller http://127.0.0.1:8888 --port 31001 --worker http://127.0.0.1:31001 --host=127.0.0.1 &
python3 -m arena.serve.model_worker --model-path claude-3-opus-20240229 --controller http://127.0.0.1:8888 --port 31016 --worker http://127.0.0.1:31016 --host=127.0.0.1 &
python3 -m arena.serve.model_worker --model-path claude-3-sonnet-20240229 --controller http://127.0.0.1:8888 --port 31019 --worker http://127.0.0.1:31019 --host=127.0.0.1 &
python3 -m arena.serve.model_worker --model-path claude-3-haiku-20240307 --controller http://127.0.0.1:8888 --port 31020 --worker http://127.0.0.1:31020 --host=127.0.0.1 &
# python3 -m arena.serve.model_worker --model-path yi-vl-plus --controller http://127.0.0.1:8888 --port 31021 --worker http://127.0.0.1:31021 --host=127.0.0.1 &
python3 -m arena.serve.model_worker --model-path Reka-Flash --controller http://127.0.0.1:8888 --port 31022 --worker http://127.0.0.1:31022 --host=127.0.0.1 &
python3 -m arena.serve.model_worker --model-path idefics2-8b-chatty --controller http://127.0.0.1:8888 --port 31023 --worker http://127.0.0.1:31023 --host=127.0.0.1 &
python3 -m arena.serve.model_worker --model-path gpt-4o --controller http://127.0.0.1:8888 --port 31024 --worker http://127.0.0.1:31024 --host=127.0.0.1 &
python3 -m arena.serve.model_worker --model-path gemini-1.5-flash-latest --controller http://127.0.0.1:8888 --port 31025 --worker http://127.0.0.1:31025 --host=127.0.0.1 &
python3 -m arena.serve.model_worker --model-path minicpm-llama3-v --controller http://127.0.0.1:8888 --port 31026 --worker http://127.0.0.1:31026 --host=127.0.0.1 &
python3 -m arena.serve.model_worker --model-path Reka-Core --controller http://127.0.0.1:8888 --port 31027 --worker http://127.0.0.1:31027 --host=127.0.0.1 &
# python3 -m arena.serve.model_worker --model-path qwen-vl-max --controller http://127.0.0.1:8888 --port 31028 --worker http://127.0.0.1:31028 --host=127.0.0.1 &
# python3 -m arena.serve.model_worker --model-path qwen-vl-plus --controller http://127.0.0.1:8888 --port 31029 --worker http://127.0.0.1:31029 --host=127.0.0.1 &
python3 -m arena.serve.model_worker --model-path gpt-4-turbo --controller http://127.0.0.1:8888 --port 31030 --worker http://127.0.0.1:31030 --host=127.0.0.1 &
python3 -m arena.serve.model_worker --model-path claude-3-5-sonnet-20240620 --controller http://127.0.0.1:8888 --port 31031 --worker http://127.0.0.1:31031 --host=127.0.0.1 &
# python3 -m arena.serve.model_worker --model-path gemini-1.5-pro-latest --controller http://127.0.0.1:8888 --port 31032 --worker http://127.0.0.1:31032 --host=127.0.0.1 &


# add llava-next-72b 
python3 -m arena.serve.model_worker --model-path gpt-4o-2024-05-13 --controller http://127.0.0.1:8888 --port 32011 --worker http://127.0.0.1:32011 --host=127.0.0.1 &
python3 -m arena.serve.model_worker --model-path gpt-4o-mini --controller http://127.0.0.1:8888 --port 32012 --worker http://127.0.0.1:32012 --host=127.0.0.1 &

```

## URL monitor

```bash
bash scripts/watch_update_url.sh
```


## Update leaderboard 

```bash
cd ~/Arena-Elo/
bash update_elo_rating_local.sh
```

## Image URL 

```bash
python img_web.py
export WEB_IMG_URL_ROOT="http://34.19.37.54:5006"
```

