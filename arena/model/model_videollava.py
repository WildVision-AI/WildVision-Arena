import torch
from icecream import ic
from arena.vlm_utils.videollava.utils import disable_torch_init

from io import BytesIO
import base64
from PIL import Image

from arena.vlm_utils.videollava.model.builder import load_pretrained_model
from arena.vlm_utils.videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from arena.vlm_utils.videollava.conversation import conv_templates, SeparatorStyle
from arena.vlm_utils.videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

@torch.inference_mode()
def generate_stream_videollava(model, tokenizer, processor, params, device, context_len, stream_interval, judge_sent_end=False):
    prompt = params["prompt"]["text"]
    
    temperature = float(params.get("temperature", 0.2))
    # top_p = float(params.get("top_p", 0.7))
    do_sample = temperature > 0.0
    max_new_tokens = min(int(params.get("max_new_tokens", 200)), 200)
        

    import json
    vision_input = BytesIO(base64.b64decode(json.loads(params["prompt"]["video"])))
    # save tmp video file
    import datetime
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    temp_file = f"/tmp/wvarena/video/{str(cur_time)}.mp4"
    import os
    if not os.path.exists(os.path.dirname(temp_file)):
        os.makedirs(os.path.dirname(temp_file))
    with open(temp_file, "wb") as output_file:
        output_file.write(vision_input.getvalue())
    vision_input = temp_file
    ic(">>> generate_stream_videollava")

    disable_torch_init()
    inp = prompt

    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    video_tensor = video_processor(vision_input, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

    print(f"{roles[1]}: {inp}")
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

    yield {"text": outputs}