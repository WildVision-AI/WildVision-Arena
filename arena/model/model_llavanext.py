import torch
from icecream import ic

from io import BytesIO
import base64
from PIL import Image
from decord import VideoReader, cpu

from arena.vlm_utils.llavavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from arena.vlm_utils.llavavid.conversation import conv_templates, SeparatorStyle
from arena.vlm_utils.llavavid.model.builder import load_pretrained_model
from arena.vlm_utils.llavavid.utils import disable_torch_init
from arena.vlm_utils.llavavid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def load_video(video_path, for_get_frames_num):
    import numpy as np
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    # sample_fps = args.for_get_frames_num if total_frame_num > args.for_get_frames_num else total_frame_num
    if len(frame_idx) > for_get_frames_num:
        sample_fps = for_get_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return spare_frames

@torch.inference_mode()
def generate_stream_llavanext(model, tokenizer, processor, params, device, context_len, stream_interval, judge_sent_end=False):
    prompt = params["prompt"]["text"]
    cur_prompt = prompt
    
    # import pdb; pdb.set_trace()
    temperature = float(params.get("temperature", 0.9))
    # top_p = float(params.get("top_p", 0.7))
    do_sample = temperature > 0.0
    max_new_tokens = min(int(params.get("max_new_tokens", 1024)), 1024)

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
    ic(">>> generate_stream_llavanext")

    disable_torch_init()
    # print(vision_input)
    vision_input = load_video(vision_input,32)
    video = processor.preprocess(vision_input, return_tensors="pt")["pixel_values"]
    video = video.to(model.device, dtype=torch.float16)
    video = [video]
    
    qs = prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv_mode = "vicuna_v1"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # print(prompt)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # print(len(input_ids))
    # import pdb; pdb.set_trace()
    with torch.inference_mode():
        # model.update_prompt([[cur_prompt]])
        output_ids = model.generate(
            inputs=input_ids.to(model.device), 
            images=video, 
            attention_mask=attention_masks.to(model.device), 
            modalities="video", 
            do_sample=do_sample, 
            temperature=temperature, 
            max_new_tokens=max_new_tokens, 
            num_beams=1,
            top_p=0.1,
            use_cache=True, 
            stopping_criteria=[stopping_criteria]
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()

    yield {"text": outputs}



@torch.inference_mode()
def generate_stream_llavanext_qwen(model, tokenizer, processor, params, device, context_len, stream_interval, judge_sent_end=False):
    prompt = params["prompt"]["text"]
    cur_prompt = prompt
    
    temperature = float(params.get("temperature", 0.9))
    # top_p = float(params.get("top_p", 0.7))
    do_sample = temperature > 0.0
    max_new_tokens = min(int(params.get("max_new_tokens", 1024)), 1024)

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
    ic(">>> generate_stream_llavanext_qwen")

    disable_torch_init()
    vision_input = load_video(vision_input,32)
    video = processor.preprocess(vision_input, return_tensors="pt")["pixel_values"]
    video = video.to(model.device, dtype=torch.float16)
    video = [video]
    
    qs = prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv_mode = "qwen_1_5"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

    # import pdb; pdb.set_trace()
    if tokenizer.pad_token_id is None:
        if "qwen" in tokenizer.name_or_path.lower():
            print("Setting pad token to bos token for qwen model.")
            tokenizer.pad_token_id = 151643

    attention_masks = input_ids.ne(tokenizer.pad_token_id).long()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    # import pdb; pdb.set_trace()
    with torch.inference_mode():
        # model.update_prompt([[cur_prompt]])

        output_ids = model.generate(
            inputs=input_ids.to(model.device), 
            images=video, 
            attention_mask=attention_masks.to(model.device), 
            modalities="video", 
            do_sample=do_sample, 
            temperature=temperature, 
            max_new_tokens=max_new_tokens, 
            use_cache=True, 
            top_p=0.1,
            num_beams=1,
            stopping_criteria=[stopping_criteria]
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()

    yield {"text": outputs}