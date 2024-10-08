"""
Chatbot Arena (side-by-side) tab.
Users chat with two chosen models.
"""

import json
import time

import gradio as gr
import numpy as np

from arena.constants import (
    LOGDIR,
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    HEADER_MD,
    INFO_MD,
    VISITBENCH_DATASETS,
    TOUCHSTONE_DATASETS,
)
from arena.model.model_adapter import get_conversation_template
from arena.serve.gradio_web_server import (
    State,
    bot_response,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
    invisible_btn,
    no_change_textbox,
    enable_textbox,
    disable_textbox,
    invisible_textbox,
    acknowledgment_md,
    get_model_description_md,
    ip_expiration_dict,
    get_ip,
)
from arena.utils import (
    build_logger,
    moderation_filter,
)

from icecream import ic

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_sides = 2
enable_moderation = False


def set_global_vars_named(enable_moderation_):
    global enable_moderation
    enable_moderation = enable_moderation_


def load_demo_side_by_side_named(models, url_params):
    # models = [model for model in models if model not in VIDEO_MODEL_LIST]
    states = (None,) * num_sides

    model_left = models[0] if len(models) > 0 else ""
    if len(models) > 1:
        weights = ([8] * 4 + [4] * 8 + [1] * 32)[: len(models) - 1]
        weights = weights / np.sum(weights)
        model_right = np.random.choice(models[1:], p=weights)
    else:
        model_right = model_left

    selector_updates = (
        gr.Dropdown(choices=models, value=model_left, visible=True),
        gr.Dropdown(choices=models, value=model_right, visible=True),
    )
    print("Side by side models", models)
    
    model_description_md_updates = (gr.Markdown(get_model_description_md(models)),)

    return states + selector_updates + model_description_md_updates


def vote_last_response(states, vote_type, reason_textbox, model_selectors, imagebox, request: gr.Request):
    # TODO: no extra save when vote; should use convide to concatenate each data
    # import os
    # input_image = imagebox
    # IMGDIR = os.path.join(LOGDIR, os.path.splitext(os.path.basename(get_conv_log_filename()))[0] + "input_images")
    # if not os.path.exists(IMGDIR): os.makedirs(IMGDIR)
    tstamp = round(time.time(), 4)
    # input_image.save(os.path.join(IMGDIR, f"input_image_{tstamp}.png"))
    
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": tstamp,
            "type": vote_type,
            "reason_textbox": reason_textbox,
            "models": [x for x in model_selectors],
            "states": [x.dict() for x in states],
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")


def leftvote_last_response(
    state0, state1, reason_textbox, model_selector0, model_selector1, imagebox, request: gr.Request
):
    logger.info(f"leftvote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "leftvote", reason_textbox, [model_selector0, model_selector1], imagebox, request
    )
    return ("",) + (disable_btn,) * 4 + (disable_textbox,)


def rightvote_last_response(
    state0, state1, reason_textbox, model_selector0, model_selector1, imagebox, request: gr.Request
):
    logger.info(f"rightvote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "rightvote", reason_textbox, [model_selector0, model_selector1], imagebox, request
    )
    return ("",) + (disable_btn,) * 4 + (disable_textbox,)


def tievote_last_response(
    state0, state1, reason_textbox, model_selector0, model_selector1, imagebox, request: gr.Request
):
    logger.info(f"tievote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "tievote", reason_textbox, [model_selector0, model_selector1], imagebox, request
    )
    return ("",) + (disable_btn,) * 4 + (disable_textbox,)


def bothbad_vote_last_response(
    state0, state1, reason_textbox, model_selector0, model_selector1, imagebox, request: gr.Request
):
    logger.info(f"bothbad_vote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "bothbad_vote", reason_textbox, [model_selector0, model_selector1], imagebox, request
    )
    return ("",) + (disable_btn,) * 4 + (disable_textbox,)


def regenerate(state0, state1, request: gr.Request):
    logger.info(f"regenerate (named). ip: {get_ip(request)}")
    states = [state0, state1]
    for i in range(num_sides):
        states[i].conv.update_last_message(None)
    return states + [x.to_gradio_chatbot() for x in states] + [""] + [disable_btn] * 6 + [disable_textbox,]


def clear_history(request: gr.Request):
    logger.info(f"clear_history (named). ip: {get_ip(request)}")
    return (
        [None] * num_sides
        + [None] * num_sides
        + [""]
        + [gr.update(value=None, interactive=True)]
        + [invisible_btn] * 4
        + [disable_btn] * 2
        + [invisible_textbox]
    )


def sample_input(request: gr.Request):
    logger.info(f"sample_input (anony_bench). ip: {get_ip(request)}")

    if random.choice([True, False]):
        sample_examle = get_random_examples_visitbench(VISITBENCH_DATASETS, num_examples=1)
    else:
        sample_examle = get_random_examples_touchstone(TOUCHSTONE_DATASETS, num_examples=1)
    image_path, text_prompt = sample_examle[0][0], sample_examle[0][1]
    return (
        [None] * num_sides
        + [None] * num_sides
        + [text_prompt]
        + [gr.update(value=image_path, interactive=True)]
        + [invisible_btn] * 4
        + [disable_btn] * 2
        + [invisible_textbox]
    )

def share_click(state0, state1, model_selector0, model_selector1, request: gr.Request):
    logger.info(f"share (named). ip: {get_ip(request)}")
    if state0 is not None and state1 is not None:
        vote_last_response(
            [state0, state1], "share", [model_selector0, model_selector1], request
        )


def add_text(
    state0, state1, model_selector0, model_selector1, text, image, request: gr.Request
):
    ip = get_ip(request)
    logger.info(f"add_text (named). ip: {ip}. len: {len(text)}")
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]

    # Init states if necessary
    for i in range(num_sides):
        if states[i] is None:
            states[i] = State(model_selectors[i])

    if len(text) <= 0:
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [""]
            + [
                no_change_btn,
            ]
            * 6
            + [no_change_textbox]
        )

    model_list = [states[i].model_name for i in range(num_sides)]
    flagged = moderation_filter(text, model_list)
    if flagged:
        logger.info(f"violate moderation (named). ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [CONVERSATION_LIMIT_MSG]
            + [
                no_change_btn,
            ]
            * 6
            + [no_change_textbox]
        )

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    # image = np.array(image, dtype='uint8').tolist()
    for i in range(num_sides):
        states[i].conv.append_message(states[i].conv.roles[0], text)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False
        states[i].conv.set_vision_input(image)
    # states += image
    return (
        states
        + [x.to_gradio_chatbot() for x in states]
        + [""]
        + [
            disable_btn,
        ]
        * 6
        + [disable_textbox]
    )


def bot_response_multi(
    state0,
    state1,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
):
    logger.info(f"bot_response_multi (named). ip: {get_ip(request)}")

    if state0.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (
            state0,
            state1,
            state0.to_gradio_chatbot(),
            state1.to_gradio_chatbot(),
        ) + (no_change_btn,) * 6 + (no_change_textbox,)
        return

    states = [state0, state1]
    gen = []
    for i in range(num_sides):
        gen.append(
            bot_response(
                states[i],
                temperature,
                top_p,
                max_new_tokens,
                request,
            )
        )

    chatbots = [None] * num_sides
    while True:
        stop = True
        for i in range(num_sides):
            try:
                ret = next(gen[i])
                states[i], chatbots[i] = ret[0], ret[1]
                stop = False
            except StopIteration:
                pass
        yield states + chatbots + [disable_btn] * 6 + [disable_textbox]
        if stop:
            break


def flash_buttons():
    btn_updates = [
        [disable_btn] * 4 + [enable_btn] * 2 + [disable_textbox],
        [enable_btn] * 6 + [enable_textbox],
    ]
    for i in range(4):
        yield btn_updates[i % 2]
        time.sleep(0.5)

import random
def get_random_examples_visitbench(dataset, num_examples=5):
    examples = []
    for idx in range(num_examples):
        example = random.choice(dataset)
        formatted_example = [example["image"], example['instruction']]
        examples.append(formatted_example)
    return examples

def get_random_examples_touchstone(dataset, num_examples=5):
    examples = []
    for idx in range(num_examples):
        example = random.choice(dataset)
        formatted_example = [example["image_input"], example['question']]
        examples.append(formatted_example)
    return examples


def build_side_by_side_ui_named(models):
    notice_markdown = f""" 
{HEADER_MD}
## 🤖 Choose two models to compare
"""
    info_markdown = f"""
    {INFO_MD}
    """
    rule_markdown = f"""
## 📜 Rules
- Upload any image, and send any question to any two models side-by-side and vote!
- Single image multi-round chat is allowed, you can continue to send question until you identify a winner.
- Click "Clear history" to start a new round.
    """
    sample_markdown = f"""
```
The **Sample Input** button aims to give you a randomly sampled example from existing benchmarks such as VisIT-Bench. 
```

## 🚶 How to use Sample Input?
- Click button "Sample Input" to randomly sample the image and text input from public benchmark to a pair of anonymous models .
- Click button "Send" and wait for the model response.
- Vote for the better one!
- Single image multi-round chat is allowed, you can continue to send question until you identify a winner.
    """
    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    chatbots = [None] * num_sides

    notice = gr.Markdown(notice_markdown, elem_id="notice_markdown")
    # models = [model for model in models if model not in VIDEO_MODEL_LIST]
    with gr.Blocks(elem_id="share-region-named"):
        

        with gr.Row():
            with gr.Accordion("🔍 Expand to see model descriptions", open=False):
                model_description_md = get_model_description_md(models)
                model_description = gr.Markdown(model_description_md, elem_id="model_description_markdown")

        with gr.Row():
            with gr.Column(scale=1):
                imagebox = gr.Image(type="pil", height=550)
                gr.Markdown(rule_markdown, elem_id="rule_markdown")
            with gr.Column(scale=2):
                with gr.Row():
                    for i in range(num_sides):
                        with gr.Column():
                            model_selectors[i] = gr.Dropdown(
                                choices=models,
                                value=models[i] if len(models) > i else "",
                                interactive=True,
                                show_label=False,
                                container=False,
                            )
                            
                with gr.Row():
                    for i in range(num_sides):
                        label = "Model A" if i == 0 else "Model B"
                        with gr.Column():
                            chatbots[i] = gr.Chatbot(
                                label=label, elem_id=f"chatbot", height=700, show_copy_button=True,
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
        reason_textbox = gr.Textbox(label="Reason", placeholder="Please input your reason for response preference here before clicking the model choice button.", type="text", elem_classes="", max_lines=5, lines=2, show_copy_button=False, visible=False, scale=2, interactive=False)

    with gr.Row():
        leftvote_btn = gr.Button(
            value="👈  A is better", visible=False, interactive=False
        )
        rightvote_btn = gr.Button(
            value="👉  B is better", visible=False, interactive=False
        )
        tie_btn = gr.Button(value="🤝  Tie", visible=False, interactive=False)
        bothbad_btn = gr.Button(
            value="👎  Both are bad", visible=False, interactive=False
        )
    with gr.Row() as button_row:
        sample_btn = gr.Button(value="🎲 Sample Input", interactive=True)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)

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
    gr.Markdown(sample_markdown, elem_id="sample_markdown")

    gr.Examples(examples=[
        ["examples/ticket.png", "Which section's ticket would you recommend I purchase?"], 
        ["examples/equation.png", "Can you derive Equation 6 from the image?"],
        ["examples/map.png", "Given my horse's location on this map, what is the quickest route to reach it?"],
        ["examples/timesquare.png", "What is the best way to commute from Trump Tower to the location shown in this image?"]
    ], inputs=[imagebox, textbox])
    gr.Markdown(info_markdown, elem_id="info_markdown")
    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    # Register listeners
    btn_list = [
        leftvote_btn,
        rightvote_btn,
        tie_btn,
        bothbad_btn,
        regenerate_btn,
        clear_btn,
    ]
    leftvote_btn.click(
        leftvote_last_response,
        states + [reason_textbox] + model_selectors + [imagebox],
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn] + [reason_textbox],
    )
    rightvote_btn.click(
        rightvote_last_response,
        states + [reason_textbox] + model_selectors + [imagebox],
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn] + [reason_textbox],
    )
    tie_btn.click(
        tievote_last_response,
        states + [reason_textbox] + model_selectors + [imagebox],
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn] + [reason_textbox],
    )
    bothbad_btn.click(
        bothbad_vote_last_response,
        states + [reason_textbox] + model_selectors + [imagebox],
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn] + [reason_textbox],
    )
    regenerate_btn.click(
        regenerate, states, states + chatbots + [textbox] + btn_list + [reason_textbox]
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list + [reason_textbox],
    ).then(
        flash_buttons, [], btn_list + [reason_textbox]
    )
    clear_btn.click(clear_history, None, states + chatbots + [textbox, imagebox] + btn_list + [reason_textbox])

    sample_btn.click(
        sample_input,
        None,
        states + chatbots + [textbox, imagebox] + btn_list + [reason_textbox],
    )
    
    for i in range(num_sides):
        model_selectors[i].change(
            clear_history, None, states + chatbots + [textbox, imagebox] + btn_list
        )

    textbox.submit(
        add_text,
        states + model_selectors + [textbox, imagebox],
        states + chatbots + [textbox] + btn_list + [reason_textbox],
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list + [reason_textbox],
    ).then(
        flash_buttons, [], btn_list + [reason_textbox]
    )
    send_btn.click(
        add_text,
        states + model_selectors + [textbox, imagebox],
        states + chatbots + [textbox] + btn_list + [reason_textbox],
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list + [reason_textbox],
    ).then(
        flash_buttons, [], btn_list + [reason_textbox]
    )
    
    # textbox.submit(
    #     add_text,
    #     states + model_selectors + [textbox, imagebox],
    #     states + chatbots + [textbox] + btn_list,
    # ).then(
    #     bot_response_multi,
    #     states + [temperature, top_p, max_output_tokens],
    #     states + chatbots + btn_list,
    # ).then(
    #     flash_buttons, [], btn_list
    # )
    # send_btn.click(
    #     add_text,
    #     states + model_selectors + [textbox, imagebox],
    #     states + chatbots + [textbox] + btn_list,
    # ).then(
    #     bot_response_multi,
    #     states + [temperature, top_p, max_output_tokens],
    #     states + chatbots + btn_list,
    # ).then(
    #     flash_buttons, [], btn_list
    # )

    return states + model_selectors + [model_description]
