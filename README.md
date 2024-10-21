# WildVision-Arena

## Enviroment
Follow [INSTALL](INSTALL.md) to install environments.

## Quick Start for Image LLM
You can quickly test model on gradio demo like below (take LLaVA as an example here):
```bash
export WILDVISION_ARENA_LOGDIR="../log"
export DOWNLOAD_DATASET=NA
python -m arena.serve.controller --host='0.0.0.0' --port 5007
python3 -m arena.serve.model_worker --model-path liuhaotian/llava-v1.5-13b --controller http://127.0.0.1:5007 --port 31002 --worker http://127.0.0.1:31002 --host=127.0.0.1  --num-gpus 1
python3 -m arena.serve.model_worker --model-path gpt-4-vision-preview --controller http://127.0.0.1:5007 --port 31001 --worker http://127.0.0.1:31001 --host=127.0.0.1 
python -m arena.serve.gradio_web_server_multi_new --share --port 5678 --controller-url http://127.0.0.1:8888 --elo-results-file ./image_elo_results.pkl --leaderboard-table-file ./image_leaderboard.csv --video-elo-results-file ./video_elo_results.pkl --video-leaderboard-table-file ./video_leaderboard.csv
```

## Quick Start for Video LLM
You can quickly test model on gradio demo like below (take Video-LLaVA as an example here):
```bash
export WILDVISION_ARENA_LOGDIR="../log"
export DOWNLOAD_DATASET=NA
python -m arena.serve.controller --host='127.0.0.1' --port 21002
python3 -m arena.serve.model_worker --model-path LanguageBind/Video-LLaVA-7B --controller http://127.0.0.1:21002 --port 32001 --worker http://127.0.0.1:32001 --host=127.0.0.1  --num-gpus 1
python -m arena.serve.gradio_web_server_multi_new --share --port 8688 --controller-url http://127.0.0.1:21002
```

Navigate to Tab `Video Arena` or `Direct Chat` to chat with Video-LLaVA. You can click video example on the left column from `Examples` to load an video question answer.


## Citation

If you found this repository useful, please consider cite our paper and resources:

```bibtex

@article{lu2024wildvision,
  title={WildVision: Evaluating Vision-Language Models in the Wild with Human Preferences},
  author={Lu, Yujie and Jiang, Dongfu and Chen, Wenhu and Wang, William Yang and Choi, Yejin and Lin, Bill Yuchen},
  publisher={NeurIPS},
  year={2024}
}
@misc{yujie2024wildvisionarena,
    title={WildVision Arena: Benchmarking Multimodal LLMs in the Wild},
    url={https://huggingface.co/spaces/WildVision/vision-arena/},
    author={Lu, Yujie and Jiang, Dongfu and Chen, Hui and Ma, Yingzi and Gu, Jing and Xiao, Chaowei and Chen, Wenhu and Wang, William and Choi, Yejin and Lin, Bill Yuchen},
    year={2024}
}
@misc{yujie2024wildvisionv2,
    title={WildVision Data and Model},
    url={https://huggingface.co/WildVision},
    author={Lu, Yujie* and Jiang, Dongfu* and Chen, Hui* and Fu, Xingyu and Ma, Yingzi and Gu, Jing and Saxon, Michael and Xiao, Chaowei and Chen, Wenhu and Choi, Yejin and Lin, Bill Yuchen and Eckstein, Miguel and Wang, William},
    year={2024}
}
```
