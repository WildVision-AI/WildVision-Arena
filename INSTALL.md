## Install Environment

```bash
# conda create -n wildvision-arena python=3.9
# conda activate wildvision-arena
# pip install -r requirements.txt
conda env create --name wildvision-arena --file=wildvision-arena.yml
conda activate wildvision-arena
pip install torchvision==0.16.2
# pip install -q -U google-generativeai
# pip install transformers-stream-generator
```

<!-- ```bash
pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e ".[model_worker,webui,train,api,dev,llm_judge]"
pip3 install transformers==4.34.0

``` -->

### Model-specific Environment

