"""
The gradio demo server for chatting with a single model.
"""

import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
import random
import time
import uuid

import gradio as gr
import requests
from icecream import ic

from arena.conversation import SeparatorStyle
from arena.constants import (
    LOGDIR,
    WORKER_API_TIMEOUT,
    ErrorCode,
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    RATE_LIMIT_MSG,
    SERVER_ERROR_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SESSION_EXPIRATION_TIME,
    RATE_LIMIT_PERIOD,
    MAX_REQUESTS_PER_DAY,
    HEADER_MD,
    INFO_MD,
    ABOUT_US_MD
)

from arena.model.model_adapter import (
    get_conversation_template,
    ANTHROPIC_MODEL_LIST,
)
from arena.model.model_registry import get_model_info, model_info
from arena.serve.api_provider import (
    anthropic_api_stream_iter,
    gemini_vision_api_stream_iter,
    idefics2_api_stream_iter,
    yivl_api_stream_iter,
    minicpm_api_stream_iter,
    qwenvl_api_stream_iter,
    reka_api_stream_iter,
    openai_api_stream_iter,
    palm_api_stream_iter,
    init_palm_chat,
)
from arena.utils import (
    build_logger,
    moderation_filter,
    get_window_url_params_js,
    get_window_url_params_with_tos_js,
    parse_gradio_auth_creds,
)

import traceback

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "FastChat Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True, visible=True)
disable_btn = gr.Button.update(interactive=False)
invisible_btn = gr.Button.update(interactive=False, visible=False)

no_change_textbox = gr.Textbox.update()
enable_textbox = gr.Textbox.update(interactive=True, visible=True)
disable_textbox = gr.Textbox.update(interactive=False)
clear_textbox = gr.Textbox.update(value="", interactive=False)
invisible_textbox = gr.Textbox.update(value="", interactive=False, visible=False)

controller_url = None
enable_moderation = False

acknowledgment_md = """

## Acknowledgment
We thank LMSYS for their great work on https://chat.lmsys.org/.
    
## Citation
```
@misc{yujie2024wildvisionarena,
    title={WildVision Arena: Benchmarking Multimodal LLMs in the Wild},
    url={https://huggingface.co/spaces/WildVision/vision-arena/},
    author={Lu, Yujie and Jiang, Dongfu and Chen, Wenhu and Wang, William and Choi, Yejin and Lin, Bill Yuchen},
    month={February},
    year={2024}
}
```
"""

ip_expiration_dict = defaultdict(lambda: 0)
# Dictionary to store request counts per IP. Format: {ip: (count, last_request_time)}
request_counts = {}

# Information about custom OpenAI compatible API models.
# JSON file format:
# {
#     "vicuna-7b": {
#         "model_name": "vicuna-7b-v1.5",
#         "api_base": "http://8.8.8.55:5555/v1",
#         "api_key": "password"
#     },
# }
openai_compatible_models_info = {}


class State:
    def __init__(self, model_name):
        self.conv = get_conversation_template(model_name)
        self.conv_id = uuid.uuid4().hex
        self.skip_next = False
        self.model_name = model_name
        

        if model_name in ["palm-2", "gemini-pro"]:
            self.palm_chat = init_palm_chat(model_name)

    def to_gradio_chatbot(self):
        return self.conv.to_gradio_chatbot()

    def dict(self):
        base = self.conv.dict()
        base.update(
            {
                "conv_id": self.conv_id,
                "model_name": self.model_name,
            }
        )
        return base


def set_global_vars(controller_url_, enable_moderation_):
    global controller_url, enable_moderation
    controller_url = controller_url_
    enable_moderation = enable_moderation_


def get_conv_log_filename():
    t = datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list(
    controller_url, register_openai_compatible_models, add_chatgpt, add_claude, add_palm
):
    if controller_url:
        # ret = requests.post(controller_url + "/refresh_all_workers")
        # assert ret.status_code == 200
        ret = requests.post(controller_url + "/list_models")
        models = ret.json()["models"]
    else:
        models = []

    # Add API providers
    if register_openai_compatible_models:
        global openai_compatible_models_info
        openai_compatible_models_info = json.load(
            open(register_openai_compatible_models)
        )
        models += list(openai_compatible_models_info.keys())

    if add_chatgpt:
        models += [
            "gpt-4-0314",
            "gpt-4-0613",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-1106",
        ]
    if add_claude:
        models += ["claude-2.1", "claude-2.0", "claude-instant-1"]
    if add_palm:
        models += ["gemini-pro"]
    models = list(set(models))

    hidden_models = ["gpt-4-0314", "gpt-4-0613"]
    for hm in hidden_models:
        if hm in models:
            del models[models.index(hm)]

    priority = {k: f"___{i:03d}" for i, k in enumerate(model_info)}
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


def load_demo_single(models, url_params):
    selected_model = models[0] if len(models) > 0 else ""
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            selected_model = model

    dropdown_update = gr.Dropdown.update(
        choices=models, value=selected_model, visible=True
    )

    state = None
    return state, dropdown_update


def load_demo(url_params, request: gr.Request):
    global models

    ip = get_ip(request)
    logger.info(f"load_demo. ip: {ip}. params: {url_params}")
    ip_expiration_dict[ip] = time.time() + SESSION_EXPIRATION_TIME

    if args.model_list_mode == "reload":
        models = get_model_list(
            controller_url,
            args.register_openai_compatible_models,
            args.add_chatgpt,
            args.add_claude,
            args.add_palm,
        )

    return load_demo_single(models, url_params)


def vote_last_response(state, vote_type, reason_textbox, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "reason": reason_textbox,
            "model": model_selector,
            "state": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, reason_textbox, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"upvote. ip: {ip}")
    vote_last_response(state, "upvote", reason_textbox, model_selector, request)
    return ("",) + (disable_btn,) * 3 + (disable_textbox,)


def downvote_last_response(state, reason_textbox, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"downvote. ip: {ip}")
    vote_last_response(state, "downvote", reason_textbox, model_selector, request)
    return ("",) + (disable_btn,) * 3 + (disable_textbox,)


def flag_last_response(state, reason_textbox, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"flag. ip: {ip}")
    vote_last_response(state, "flag", reason_textbox, model_selector, request)
    return ("",) + (disable_btn,) * 3 + (disable_textbox,)


def regenerate(state, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"regenerate. ip: {ip}")
    state.conv.update_last_message(None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5 + (disable_textbox,)


def clear_history(request: gr.Request):
    ip = get_ip(request)
    logger.info(f"clear_history. ip: {ip}")
    state = None
    return (state, [], "", gr.update(value=None, interactive=True)) + (disable_btn,) * 5 + (clear_textbox,)


def get_ip(request: gr.Request):
    if "cf-connecting-ip" in request.headers:
        ip = request.headers["cf-connecting-ip"]
    else:
        ip = request.client.host
    return ip


def add_text(state, model_selector, text, image, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"add_text. ip: {ip}. len: {len(text)}")

    if state is None:
        state = State(model_selector)

    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "") + (no_change_btn,) * 5 + (no_change_textbox,)

    flagged = moderation_filter(text, [state.model_name])
    if flagged:
        logger.info(f"violate moderation. ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    conv = state.conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), CONVERSATION_LIMIT_MSG) + (
            no_change_btn,
        ) * 5 + (no_change_textbox,)

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    logger.info(f"type image{type(image)}")
    conv.set_image(image)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5 + (disable_textbox,)


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


def model_worker_stream_iter(
    conv,
    model_name,
    worker_addr,
    prompt,
    temperature,
    repetition_penalty,
    top_p,
    max_new_tokens,
):
    # Make requests
    # TODO: trans state
    import base64
    from io import BytesIO
    from PIL import Image

    image = conv.get_image()
    
    from icecream import ic
    im_file = BytesIO()
    image.save(im_file, format="PNG") # TODO: fix; when not read from file, no format, default PNG
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes)

    ic(">>> send params")
    gen_params = {
        "model": model_name,
        "prompt": {"text":prompt, "image": json.dumps(im_b64.decode("utf-8"))},
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }
    input_text = gen_params["prompt"]["text"]
    logger.info(f"==== model worker stream iter request ====\n{input_text}")

    # Stream output
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json=gen_params,
        stream=True,
        timeout=WORKER_API_TIMEOUT,
    )
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            yield data

def request_allowed(ip):
    """
    Check if the request from the given IP is allowed based on the rate limit.

    Args:
    - ip (str): The IP address of the incoming request.

    Returns:
    - bool: True if the request is allowed, False otherwise.
    """
    current_time = datetime.now()
    if ip in request_counts:
        count, last_request_time = request_counts[ip]
        logger.info(f"rate_limit_per_day. ip: {ip} count: {count}")
        if current_time - last_request_time < RATE_LIMIT_PERIOD:
            if count < MAX_REQUESTS_PER_DAY:
                # Increment count and update the time for the current request
                request_counts[ip] = (count + 1, current_time)
                return True
            else:
                # Rate limit exceeded
                logger.info(f"rate_limit_per_day exceeded. ip: {ip} count: {count} limit: {MAX_REQUESTS_PER_DAY}")
                return False
        else:
            # Reset count after a day has passed
            request_counts[ip] = (1, current_time)
            return True
    else:
        # First request from this IP
        request_counts[ip] = (1, current_time)
        return True


def bot_response(
    state,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
    apply_rate_limit=True,
):
    ip = get_ip(request)
    logger.info(f"bot_response. ip: {ip}")

    start_tstamp = time.time()
    temperature = float(temperature)
    top_p = float(top_p)
    max_new_tokens = int(max_new_tokens)
    
    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        state.skip_next = False
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5 + (no_change_textbox,)
        return

    conv, model_name = state.conv, state.model_name
    logger.info(f"conv: {conv}")

    ic(model_name)
    image = conv.get_image()
    if model_name in [
        "gpt-4-vision-preview", "gpt-4o", "gpt-4-turbo",
    ]:
        prompt = conv.to_openai_api_messages()
        stream_iter = openai_api_stream_iter(
            model_name, prompt, temperature, top_p, max_new_tokens, image
        )
    elif model_name in [
        "gemini-pro-vision", "gemini-1.5-flash-latest",
    ]:
        stream_iter = gemini_vision_api_stream_iter(
            model_name, conv.messages[-2][1], temperature, top_p, max_new_tokens, image
        )
    elif model_name in [
        "idefics2-8b", "idefics2-8b-chatty",
    ]:
        stream_iter = idefics2_api_stream_iter(
            model_name, conv, temperature, top_p, max_new_tokens, image
        )
    elif model_name in [
        "yi-vl-plus",
    ]:
        # TODO: double check yi-vl-api template and history
        prompt = conv.to_openai_api_messages()
        stream_iter = yivl_api_stream_iter(
            model_name, prompt, temperature, top_p, max_new_tokens, image
        )
    elif model_name in [
        "mini-cpm-llama3-v-2.5", "minicpm-llama3-v"
    ]:
        stream_iter = minicpm_api_stream_iter(
            model_name, conv, temperature, top_p, max_new_tokens, image
        )
    elif model_name in [
        "qwen-vl-plus", "qwen-vl-max", "qwen2-72b-instruct"
    ]:
        stream_iter = qwenvl_api_stream_iter(
            model_name, conv, temperature, top_p, max_new_tokens, image
        )
    elif model_name in [
        "reka",
        "Reka-Flash",
        "creeping-phlox-20240403",
        "Reka-Core",
    ]:
        # TODO: double check reka template and history
        # TODO: add reka conversation history: https://docs.reka.ai/guides/002-image-video-audio-chat.html
        
        if model_name == "Reka-Flash":
            model_name = "creeping-phlox-20240403" # For WildVision Arena use
        stream_iter = reka_api_stream_iter(
            model_name, conv, temperature, top_p, max_new_tokens, image
        )
    elif model_name in openai_compatible_models_info:
        model_info = openai_compatible_models_info[model_name]
        prompt = conv.to_openai_api_messages()
        stream_iter = openai_api_stream_iter(
            model_info["model_name"],
            prompt,
            temperature,
            top_p,
            max_new_tokens,
            api_base=model_info["api_base"],
            api_key=model_info["api_key"],
        )
    elif model_name in [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-1106",
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-0613",
        "gpt-4-turbo",
    ]:
        # avoid conflict with Azure OpenAI
        assert model_name not in openai_compatible_models_info
        prompt = conv.to_openai_api_messages()
        stream_iter = openai_api_stream_iter(
            model_name, prompt, temperature, top_p, max_new_tokens
        )
    elif model_name in ANTHROPIC_MODEL_LIST:
        prompt = conv.get_prompt()
        stream_iter = anthropic_api_stream_iter(
            model_name, prompt, temperature, top_p, max_new_tokens, image
        )
    elif model_name in ["palm-2", "gemini-pro"]:
        stream_iter = palm_api_stream_iter(
            model_name,
            state.palm_chat,
            conv.messages[-2][1],
            temperature,
            top_p,
            max_new_tokens,
        )
    else:
        # Query worker address
        ret = requests.post(
            controller_url + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

        # No available worker
        if worker_addr == "":
            conv.update_last_message(SERVER_ERROR_MSG)
            yield (
                state,
                state.to_gradio_chatbot(),
                disable_btn,
                disable_btn,
                disable_btn,
                enable_btn,
                enable_btn,
                disable_textbox,
            )
            return

        # Construct prompt.
        # We need to call it here, so it will not be affected by "▌".
        prompt = conv.get_prompt()

        # Set repetition_penalty
        if "t5" in model_name:
            repetition_penalty = 1.2
        else:
            repetition_penalty = 1.0

        stream_iter = model_worker_stream_iter(
            conv,
            model_name,
            worker_addr,
            prompt,
            temperature,
            repetition_penalty,
            top_p,
            max_new_tokens,
        )

    conv.update_last_message("▌")
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5 + (disable_textbox,)

    try:
        if apply_rate_limit and model_name in ["gpt-4-vision-preview", "gpt-4o", "gpt-4-turbo"] and not request_allowed(ip):
            logger.info(f"reach rate_limit_per_day. ip: {ip}")
            output =f"{RATE_LIMIT_MSG}\n\n (error_code: {ErrorCode.RATE_LIMIT})"
            conv.update_last_message(output)
            yield (state, state.to_gradio_chatbot()) + (
                disable_btn,
                disable_btn,
                disable_btn,
                enable_btn,
                enable_btn,
            ) + (disable_textbox,)
            return
        else:
            for i, data in enumerate(stream_iter):
                # if data["error_code"] == 0:
                if data.get("error_code", 0) == 0:
                    output = data["text"].strip()
                    conv.update_last_message(output + "▌")
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5 + (disable_textbox,)
                else:
                    output = data["text"] + f"\n\n(error_code: {data['error_code']})"
                    conv.update_last_message(output)
                    yield (state, state.to_gradio_chatbot()) + (
                        disable_btn,
                        disable_btn,
                        disable_btn,
                        enable_btn,
                        enable_btn,
                    ) + (disable_textbox,)
                    return
        
        output = data["text"].strip()
        if "vicuna" in model_name:
            output = post_process_code(output)
        conv.update_last_message(output)
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5 + (enable_textbox,)
    except requests.exceptions.RequestException as e:
        error_info = traceback.format_exc()
        logger.info(f"An GRADIO_REQUEST_ERROR occurred: {e}. Traceback: {error_info}")
        
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n"
            f"(error_code: {ErrorCode.GRADIO_REQUEST_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        ) + (disable_textbox,)
        return
    except Exception as e:    
        error_info = traceback.format_exc()
        logger.info(f"An GRADIO_STREAM_UNKNOWN_ERROR occurred: {e}. Traceback: {error_info}")
    
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n"
            f"(error_code: {ErrorCode.GRADIO_STREAM_UNKNOWN_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        ) + (disable_textbox,)
        return

    finish_tstamp = time.time()
    logger.info(f"{output}")
    # TODO: uncomment to save image for each conversation
    input_image = conv.get_image()
    IMGDIR = os.path.join(LOGDIR, os.path.splitext(os.path.basename(get_conv_log_filename()))[0] + "input_images")
    if not os.path.exists(IMGDIR): os.makedirs(IMGDIR)
    # input_image.save(os.path.join(IMGDIR, f"input_image_{int(finish_tstamp)}.png"))
    input_image.save(os.path.join(IMGDIR, f"input_image_{state.conv_id}_{round(finish_tstamp, 4)}.png"))

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "gen_params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
            },
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")


block_css = """
#arena-tab-button{
    font-size: 15pt;
    font-weight: bold;
}
#notice_markdown {
    font-size: 110%
}
#notice_markdown th {
    display: none;
}
#notice_markdown td {
    padding-top: 6px;
    padding-bottom: 6px;
}
#model_description_markdown {
    font-size: 110%
}
#leaderboard_markdown {
    font-size: 110%
}
#leaderboard_markdown td {
    padding-top: 6px;
    padding-bottom: 6px;
}
#leaderboard_dataframe td {
    line-height: 0.1em;
}
#about_markdown {
    font-size: 110%
}
#ack_markdown {
    font-size: 110%
}
#input_box textarea {
}
footer {
    display:none !important
}
.image-container {
    display: flex;
    align-items: center;
    padding: 1px;
}
.image-container img {
    display: block;
    height: 100%;
    image-orientation: 0deg;
    max-height: none !important;
    max-width: none !important;
    min-height: 0 !important;
    min-width: 0 !important;
    width: 100%;
}
#output_box {
    font-size: 110%
    color: gray;
}
"""


def get_model_description_md(models):
    model_description_md = """
| | | |
| ---- | ---- | ---- |
"""
    ct = 0
    visited = set()
    for i, name in enumerate(models):
        minfo = get_model_info(name)
        if minfo.simple_name in visited:
            continue
        visited.add(minfo.simple_name)
        one_model_md = f"[{minfo.simple_name}]({minfo.link}): {minfo.description}"

        if ct % 3 == 0:
            model_description_md += "|"
        model_description_md += f" {one_model_md} |"
        if ct % 3 == 2:
            model_description_md += "\n"
        ct += 1
    return model_description_md


def build_about():
    about_markdown = ABOUT_US_MD

    # state = gr.State()
    gr.Markdown(about_markdown, elem_id="about_markdown")

    # return [state]


def build_single_model_ui(models, add_promotion_links=False):
    promotion = (
        f"""
{HEADER_MD}

## 🤖 Choose any model to chat
"""
        if add_promotion_links
        else ""
    )

    info_markdown = f"""
    {INFO_MD}
    """

    notice_markdown = f"""
{promotion}
"""

    state = gr.State()
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Box(elem_id="share-region-named", css = ".output-image, .input-image, .image-preview {height: 600px !important} "):
        with gr.Row(elem_id="model_selector_row"):
            model_selector = gr.Dropdown(
                choices=models,
                value=models[0] if len(models) > 0 else "",
                interactive=True,
                show_label=False,
                container=False,
            )
        with gr.Row():
            with gr.Accordion(
                "🔍 Expand to see model descriptions",
                open=False,
                elem_id="model_description_accordion",
            ):
                model_description_md = get_model_description_md(models)
                gr.Markdown(model_description_md, elem_id="model_description_markdown")
        with gr.Row():
            imagebox = gr.Image(type="pil", height=600)
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Scroll down and start chatting",
                height=600,
                show_copy_button=True,
            )
    with gr.Row():
        textbox = gr.Textbox(
            lines=2,
            show_label=False,
            placeholder="👉 Enter your prompt and press ENTER",
            container=False,
            elem_id="input_box",
            max_lines=5
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row():
        reason_textbox = gr.Textbox(label="Reason", placeholder="Please input your reason for voting here before clicking the vote button.", type="text", elem_classes="", max_lines=5, lines=2, show_copy_button=False, visible=False, scale=2, interactive=False)
            
    with gr.Row() as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    with gr.Accordion("⚙️ Parameters", open=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=2048,
            value=1024,
            step=64,
            interactive=True,
            label="Max output tokens",
        )
    gr.Examples(examples=[
        ["examples/ticket.png", "Which section's ticket would you recommend I purchase?"], 
        ["examples/equation.png", "Can you derive Equation 6 from the image?"],
        ["examples/map.png", "Given my horse's location on this map, what is the quickest route to reach it?"],
        ["examples/timesquare.png", "What is the best way to commute from Trump Tower to the location shown in this image?"]
    ], inputs=[imagebox, textbox])
    gr.Markdown(INFO_MD, elem_id="info_markdown")
    if add_promotion_links:
        gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    # Register listeners
    btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
    upvote_btn.click(
        upvote_last_response,
        [state, reason_textbox, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn, reason_textbox],
    )
    downvote_btn.click(
        downvote_last_response,
        [state, reason_textbox, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn, reason_textbox],
    )
    flag_btn.click(
        flag_last_response,
        [state, reason_textbox, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn, reason_textbox],
    )
    regenerate_btn.click(regenerate, state, [state, chatbot, textbox] + btn_list + [reason_textbox]).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list + [reason_textbox],
    )
    clear_btn.click(clear_history, None, [state, chatbot, textbox, imagebox] + btn_list + [reason_textbox])

    model_selector.change(clear_history, None, [state, chatbot, textbox, imagebox] + btn_list + [reason_textbox])

    
    textbox.submit(
        add_text, [state, model_selector, textbox, imagebox], [state, chatbot, textbox] + btn_list + [reason_textbox]
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list + [reason_textbox],
    )
    send_btn.click(
        add_text,
        [state, model_selector, textbox, imagebox],
        [state, chatbot, textbox] + btn_list + [reason_textbox],
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list + [reason_textbox],
    )

    return [state, model_selector]


def build_demo(models):
    with gr.Blocks(
        title="Chat with Open Multimodal Large Language Models",
        theme=gr.themes.Default(),
        css=block_css,
    ) as demo:
        url_params = gr.JSON(visible=False)

        state, model_selector = build_single_model_ui(models)

        if args.model_list_mode not in ["once", "reload"]:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

        if args.show_terms_of_use:
            load_js = get_window_url_params_with_tos_js
        else:
            load_js = get_window_url_params_js

        demo.load(
            load_demo,
            [url_params],
            [
                state,
                model_selector,
            ],
            _js=load_js,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to generate a public, shareable link",
    )
    parser.add_argument(
        "--controller-url",
        type=str,
        default="http://localhost:21001",
        help="The address of the controller",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=10,
        help="The concurrency count of the gradio queue",
    )
    parser.add_argument(
        "--model-list-mode",
        type=str,
        default="once",
        choices=["once", "reload"],
        help="Whether to load the model list once or reload the model list every time",
    )
    parser.add_argument(
        "--moderate",
        action="store_true",
        help="Enable content moderation to block unsafe inputs",
    )
    parser.add_argument(
        "--show-terms-of-use",
        action="store_true",
        help="Shows term of use before loading the demo",
    )
    parser.add_argument(
        "--add-chatgpt",
        action="store_true",
        help="Add OpenAI's ChatGPT models (gpt-3.5-turbo, gpt-4)",
    )
    parser.add_argument(
        "--add-claude",
        action="store_true",
        help="Add Anthropic's Claude models (claude-2, claude-instant-1)",
    )
    parser.add_argument(
        "--add-palm",
        action="store_true",
        help="Add Google's PaLM model (PaLM 2 for Chat: chat-bison@001)",
    )

    parser.add_argument(
        "--register-openai-compatible-models",
        type=str,
        help="Register custom OpenAI API compatible models by loading them from a JSON file",
    )
    parser.add_argument(
        "--gradio-auth-path",
        type=str,
        help='Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"',
    )
    parser.add_argument(
        "--gradio-root-path",
        type=str,
        help="Sets the gradio root path, eg /abc/def. Useful when running behind a reverse-proxy or at a custom URL path prefix",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Set global variables
    set_global_vars(args.controller_url, args.moderate)
    models = get_model_list(
        args.controller_url,
        args.register_openai_compatible_models,
        args.add_chatgpt,
        args.add_claude,
        args.add_palm,
    )

    # Set authorization credentials
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)

    # Launch the demo
    demo = build_demo(models)
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
        auth=auth,
        root_path=args.gradio_root_path,
    )