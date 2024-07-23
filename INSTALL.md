## Install Environment
Create conda enviroment.
```bash
conda create -n wildvision-arena python=3.9
conda activate wildvision-arena
# pip install -r requirements.txt
# conda env create --name wildvision-arena --file=wildvision-arena.yml
# pip install torchvision==0.16.2
# pip install -q -U google-generativeai
# pip install transformers-stream-generator
```
Install packages.
```bash
pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e ".[model_worker,vision_arena,webui,api,video_arena,dev]"
pip3 install gradio==4.37.2
pip3 install gradio_client==1.0.2
# pip3 install transformers==4.34.0

```

### Model-specific Environment
**Video-LLaVA**
Follow [Video-LLaVA Install](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/pyproject.toml) for package version details.
```bash
conda create -n arena-videollava --clone wildvision-arena
conda activate arena-videollava
pip install decord opencv-python git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
```

**LLaVA-NEXT-VIDEO**
```bash
conda create -n arena-llavanextvideo --clone wildvision-arena
conda activate arena-llavanextvideo
pip install transformers@git+https://github.com/huggingface/transformers.git@1c39974a4c4036fd641bc1191cc32799f85715a4
```

**VideoLLaMA2**
```bash
conda create -n arena-videollama --clone wildvision-arena
conda activate arena-videollama
pip install transformers==4.41.2
pip install torch==2.2.0 torchvision==0.17.0 wrapt==1.14.0 moviepy scenedetect==0.6.3 opencv-python==4.7.0.72 pysubs2 bitsandbytes==0.43.0

```

### Local Test
```bash
python -m arena.serve.gradio_web_server_multi_new --port 5679 --controller-url http://0.0.0.0:8888
```