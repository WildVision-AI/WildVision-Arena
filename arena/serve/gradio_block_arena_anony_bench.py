"""
Chatbot Arena (battle) tab.
Users chat with two anonymous models.
"""

import json
import time

import gradio as gr
import numpy as np

import os
import uuid
import datetime
from PIL import Image

from arena.constants import (
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    SLOW_MODEL_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    CONVERSATION_SAVE_DIR,
    SAMPLING_WEIGHTS,
    BATTLE_TARGETS,
    SAMPLING_BOOST_MODELS,
    OUTAGE_MODELS,
    HEADER_MD,
    INFO_MD,
    VISITBENCH_DATASETS,
    TOUCHSTONE_DATASETS,
)
from arena.model.model_adapter import get_conversation_template
from arena.serve.gradio_block_arena_named import flash_buttons
from arena.serve.gradio_web_server import (
    State,
    bot_response,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
    invisible_btn,
    acknowledgment_md,
    ip_expiration_dict,
    get_ip,
    get_model_description_md,
)
from arena.utils import (
    build_logger,
    moderation_filter,
)
from arena.serve.gradio_web_server import block_css

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_sides = 2
enable_moderation = False
anony_names = ["", ""]
models = []


def set_global_vars_anony_bench(enable_moderation_):
    global enable_moderation
    enable_moderation = enable_moderation_


def load_demo_side_by_side_anony_bench(models_, url_params):
    global models
    models = models_

    states = (None,) * num_sides
    selector_updates = (
        gr.Markdown(visible=True),
        gr.Markdown(visible=True),
    )

    model_description_md_updates = (gr.Markdown(get_model_description_md(models)),)
    
    return states + selector_updates + model_description_md_updates


def vote_last_response(states, vote_type, model_selectors, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "models": [x for x in model_selectors],
            "states": [x.dict() for x in states],
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")

    if ":" not in model_selectors[0]:
        for i in range(15):
            names = (
                "### Model A: " + states[0].model_name,
                "### Model B: " + states[1].model_name,
            )
            yield names + ("",) + (disable_btn,) * 4
            time.sleep(0.2)
    else:
        names = (
            "### Model A: " + states[0].model_name,
            "### Model B: " + states[1].model_name,
        )
        yield names + ("",) + (disable_btn,) * 4



def save_vote_data(state, request: gr.Request):
    t = datetime.datetime.now()
    # save image
    img_name = os.path.join(CONVERSATION_SAVE_DIR, 'images', f"{t.year}-{t.month:02d}-{t.day:02d}-{str(uuid.uuid4())}.png")
    while os.path.exists(img_name):
        img_name = os.path.join(CONVERSATION_SAVE_DIR, 'images', f"{t.year}-{t.month:02d}-{t.day:02d}-{str(uuid.uuid4())}.png")
    image = np.array(state['image'], dtype='uint8')
    image = Image.fromarray(image.astype('uint8')).convert('RGB')
    image.save(img_name)
    # save conversation
    log_name = os.path.join(CONVERSATION_SAVE_DIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conversation.json")
    with open(log_name, 'a') as fout:
        log_data = state.copy()
        log_data['image'] = img_name
        log_data['ip'] = request.client.host
        fout.write(json.dumps(state) + "\n")

def leftvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"leftvote (anony_bench). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "leftvote", [model_selector0, model_selector1], request
    ):
        yield x


def rightvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"rightvote (anony_bench). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "rightvote", [model_selector0, model_selector1], request
    ):
        yield x


def tievote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"tievote (anony_bench). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "tievote", [model_selector0, model_selector1], request
    ):
        yield x


def bothbad_vote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"bothbad_vote (anony_bench). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "bothbad_vote", [model_selector0, model_selector1], request
    ):
        yield x


def regenerate(state0, state1, request: gr.Request):
    logger.info(f"regenerate (anony_bench). ip: {get_ip(request)}")
    states = [state0, state1]
    for i in range(num_sides):
        states[i].conv.update_last_message(None)
    return states + [x.to_gradio_chatbot() for x in states] + [""] + [disable_btn] * 6


def clear_history(request: gr.Request):
    logger.info(f"clear_history (anony_bench). ip: {get_ip(request)}")
    return (
        [None] * num_sides
        + [None] * num_sides
        + anony_names
        + [""]
        + [gr.update(value=None, interactive=True)]
        + [invisible_btn] * 4
        + [disable_btn] * 2
        + [""]
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
        + anony_names
        + [text_prompt]
        + [gr.update(value=image_path, interactive=True)]
        + [invisible_btn] * 4
        + [disable_btn] * 2
        + [""]
    )

def share_click(state0, state1, model_selector0, model_selector1, request: gr.Request):
    logger.info(f"share (anony_bench). ip: {get_ip(request)}")
    if state0 is not None and state1 is not None:
        vote_last_response(
            [state0, state1], "share", [model_selector0, model_selector1], request
        )




def get_sample_weight(model):
    if model in OUTAGE_MODELS:
        return 0
    if model not in SAMPLING_WEIGHTS:
        logger.warning(f"model {model} not in SAMPLING_WEIGHTS")
    weight = SAMPLING_WEIGHTS.get(model, 1.0)
    if model in SAMPLING_BOOST_MODELS:
        weight *= 5
    return weight


def get_battle_pair():
    if len(models) == 1:
        return models[0], models[0]
    model_weights = []
    for model in models:
        weight = get_sample_weight(model)
        model_weights.append(weight)
    total_weight = np.sum(model_weights)
    model_weights = model_weights / total_weight
    chosen_idx = np.random.choice(len(models), p=model_weights)
    chosen_model = models[chosen_idx]
    # for p, w in zip(models, model_weights):
    #     print(p, w)

    rival_models = []
    rival_weights = []
    for model in models:
        if model == chosen_model:
            continue
        weight = get_sample_weight(model)
        if (
            weight != 0
            and chosen_model in BATTLE_TARGETS
            and model in BATTLE_TARGETS[chosen_model]
        ):
            # boost to 50% chance
            weight = total_weight / len(BATTLE_TARGETS[chosen_model])
        rival_models.append(model)
        rival_weights.append(weight)
    # for p, w in zip(rival_models, rival_weights):
    #     print(p, w)
    rival_weights = rival_weights / np.sum(rival_weights)
    rival_idx = np.random.choice(len(rival_models), p=rival_weights)
    rival_model = rival_models[rival_idx]

    swap = np.random.randint(2)
    if swap == 0:
        return chosen_model, rival_model
    else:
        return rival_model, chosen_model


def add_text(
    state0, state1, model_selector0, model_selector1, text, image, request: gr.Request
):
    ip = get_ip(request)
    logger.info(f"add_text (anony_bench). ip: {ip}. len: {len(text)}")
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]

    # Init states if necessary
    if states[0] is None:
        assert states[1] is None

        model_left, model_right = get_battle_pair()
        states = [
            State(model_left),
            State(model_right),
        ]

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
            + [""]
        )

    model_list = [states[i].model_name for i in range(num_sides)]
    flagged = moderation_filter(text, model_list)
    if flagged:
        logger.info(f"violate moderation (anony_bench). ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {get_ip(request)}. text: {text}")
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
            + [""]
        )

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_sides):
        states[i].conv.append_message(states[i].conv.roles[0], text)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False
        states[i].conv.set_vision_input(image)

    slow_model_msg = ""
    for i in range(num_sides):
        if "deluxe" in states[i].model_name:
            slow_model_msg = SLOW_MODEL_MSG
    return (
        states
        + [x.to_gradio_chatbot() for x in states]
        + [""]
        + [
            disable_btn,
        ]
        * 6
        + [slow_model_msg]
    )


def bot_response_multi(
    state0,
    state1,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
):
    logger.info(f"bot_response_multi (anony_bench). ip: {get_ip(request)}")

    if state0 is None or state0.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (
            state0,
            state1,
            state0.to_gradio_chatbot(),
            state1.to_gradio_chatbot(),
        ) + (no_change_btn,) * 6
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
                apply_rate_limit=False,
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
        yield states + chatbots + [disable_btn] * 6
        if stop:
            break

import random
def get_random_examples_mmmu(dataset, num_examples=5):
    examples = []
    for idx in range(num_examples):
        example = random.choice(dataset)
        if example["question_type"] != "multiple-choice": continue
        # Format the example as needed, here just a placeholder
        os.makedirs("random", exist_ok=True)
        import shortuuid
        uid = shortuuid.uuid()
        image_path = f"random/tmp_image_{uid}.png"
        example['image_1'].save(image_path)

        formatted_example = [image_path, example['question'], f"Options:{example['options']} Answer: {example['answer']}"]
        examples.append(formatted_example)
    return examples

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


def build_side_by_side_ui_anony_bench(models):
    notice_markdown = f"""
# ⚔️  WildVision Arena ⚔️ : Benchmarking Multimodal LLMs in the Wild
by WildVision Team @ AI2 + UCSB + UWaterloo

```
This tab aims to test a pair of anonymous models on a randomly sampled example from existing multimodal benchmarks such as VisIT-Bench. 
```

## 🚶 How to use Sample Input?
- Click button "Sample Input" to randomly sample the image and text input from public benchmark to a pair of anonymous models .
- Click button "Send" and wait for the model response.
- Vote for the better one!
- Single image multi-round chat is allowed, you can continue to send question until you identify a winner.

## 👇 Chat now!

    """

    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    chatbots = [None] * num_sides

    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Blocks(elem_id="share-region-anony", css=block_css):
        with gr.Accordion("🔍 Expand to see Arena players", open=False):
            model_description_md = get_model_description_md(models)
            model_description = gr.Markdown(model_description_md, elem_id="model_description_markdown")
        with gr.Row():
            imagebox = gr.Image(type="pil", height=550)
            for i in range(num_sides):
                label = "Model A" if i == 0 else "Model B"
                with gr.Column():
                    chatbots[i] = gr.Chatbot(
                        label=label, elem_id=f"chatbot", height=550, show_copy_button=True,
                    )

        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    model_selectors[i] = gr.Markdown(anony_names[i])
        with gr.Row():
            slow_warning = gr.Markdown("", elem_id="notice_markdown")

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
    # output_textbox = gr.Textbox(
    #     lines=2,
    #     show_label=False,
    #     interactive=False,
    #     placeholder="Output Reference will be shown here after response sampled.",
    #     container=False,
    #     elem_id="output_box",
    #     max_lines=2
    # )
    with gr.Row():
        textbox = gr.Textbox(
            lines=5,
            show_label=False,
            placeholder="👉 Sample or enter your prompt and press ENTER",
            container=False,
            elem_id="input_box",
            max_lines=5,
            autoscroll=True,
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row() as button_row:

        sample_btn = gr.Button(value="🎲 Sample Input", interactive=True)
        clear_btn = gr.Button(value="🎲 New Round", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        share_btn = gr.Button(value="📷  Share")

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
    
    # from datasets import load_dataset
    # subset_category = random.choice(['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology'])
    # mmmu_datasets = load_dataset("MMMU/MMMU", subset_category, split="validation")
    # random_examples = get_random_examples(mmmu_datasets, num_examples=5)
    
    # gr.Examples(examples=[
    #     ["examples/ticket.png", "Which section's ticket would you recommend I purchase?"],
    #     ["examples/swimming.png", "What swimming style is being depicted?"],
    #     ["examples/equation.png", "Can you derive Equation 6 from the image?"],
    #     ["examples/map.png", "Given my horse's location on this map, what is the quickest route to reach it?"],
    #     ["examples/timesquare.png", "What is the best way to commute from Trump Tower to the location shown in this image?"]
    # ], inputs=[imagebox, textbox])
    # gr.Examples(examples=random_examples, inputs=[imagebox, textbox])
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
        states + model_selectors,
        model_selectors + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    rightvote_btn.click(
        rightvote_last_response,
        states + model_selectors,
        model_selectors + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    tie_btn.click(
        tievote_last_response,
        states + model_selectors,
        model_selectors + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    bothbad_btn.click(
        bothbad_vote_last_response,
        states + model_selectors,
        model_selectors + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    regenerate_btn.click(
        regenerate, states, states + chatbots + [textbox] + btn_list
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )
    clear_btn.click(
        clear_history,
        None,
        states + chatbots + model_selectors + [textbox, imagebox] + btn_list + [slow_warning],
    )
    sample_btn.click(
        sample_input,
        None,
        states + chatbots + model_selectors + [textbox, imagebox] + btn_list + [slow_warning],
    )

    share_js = """
function (a, b, c, d) {
    const captureElement = document.querySelector('#share-region-anony');
    html2canvas(captureElement)
        .then(canvas => {
            canvas.style.display = 'none'
            document.body.appendChild(canvas)
            return canvas
        })
        .then(canvas => {
            const image = canvas.toDataURL('image/png')
            const a = document.createElement('a')
            a.setAttribute('download', 'chatbot-arena.png')
            a.setAttribute('href', image)
            a.click()
            canvas.remove()
        });
    return [a, b, c, d];
}
"""
    share_btn.click(share_click, states + model_selectors, [], _js=share_js)

    textbox.submit(
        add_text,
        states + model_selectors + [textbox, imagebox],
        states + chatbots + [textbox] + btn_list + [slow_warning],
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons,
        [],
        btn_list,
    )

    send_btn.click(
        add_text,
        states + model_selectors + [textbox, imagebox],
        states + chatbots + [textbox] + btn_list,
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )

    return states + model_selectors + [model_description]
