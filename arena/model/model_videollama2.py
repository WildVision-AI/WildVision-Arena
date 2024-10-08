import torch
from icecream import ic

from io import BytesIO
import base64
from PIL import Image
import transformers

from arena.vlm_utils.videollama2.conversation import conv_templates
from arena.vlm_utils.videollama2.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from arena.vlm_utils.videollama2.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, process_video, process_image
from arena.vlm_utils.videollama2.model.builder import load_pretrained_model



@torch.inference_mode()
def generate_stream_videollama2(model, tokenizer, processor, params, device, context_len, stream_interval, judge_sent_end=False):
    prompt = params["prompt"]["text"]
    prompt = prompt.strip("[INST]").strip("[/INST]").strip(" ")

    temperature = float(params.get("temperature", 0.2))
    # top_p = float(params.get("top_p", 0.7))
    do_sample = temperature > 0.0
    max_new_tokens = min(int(params.get("max_new_tokens", 200)), 200)
        
    import json
    vision_input = BytesIO(base64.b64decode(json.loads(params["prompt"]["video"])))

    import datetime
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    temp_file = f"/tmp/wvarena/video/{str(cur_time)}.mp4"
    import os
    if not os.path.exists(os.path.dirname(temp_file)):
        os.makedirs(os.path.dirname(temp_file))
    with open(temp_file, "wb") as output_file:
        output_file.write(vision_input.getvalue())
    vision_input = temp_file
    
    ic(">>> generate_stream_videollama2")

    modal_list = ['video']

    tensor = process_video(vision_input, processor, model.config.image_aspect_ratio).to(dtype=torch.float16, device='cuda', non_blocking=True)
    default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
    modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
    
    tensor = [tensor]

    conv_mode = 'llama_2'
    question = default_mm_token + "\n" + prompt
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_token_index, return_tensors='pt').unsqueeze(0).to('cuda:0')

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images_or_videos=tensor,
            modal_list=modal_list,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    yield {"text": outputs[0]}