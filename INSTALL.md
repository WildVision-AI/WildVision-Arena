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
pip3 install -e ".[model_worker,vision_arena,webui,api]"
pip3 install gradio==4.37.2
pip3 install gradio_client==1.0.2
# pip3 install transformers==4.34.0

```

### Model-specific Environment

