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

import random
random.seed(42)
import gradio as gr

all_failure_cases = json.load(open('examples/failure_cases.json'))
failure_cases_images_folder = 'examples/failure_cases_images'
failure_cases = [v for k, v in all_failure_cases.items() if v['image'] in os.listdir(failure_cases_images_folder)]

def sample_case():
    failure_case = random.choice(failure_cases)
    image_path = os.path.join(failure_cases_images_folder, failure_case['image'])
    model_a = failure_case['model_a']
    model_b = failure_case['model_b']
    conversation_a = failure_case['conversation_a']
    conversation_b = failure_case['conversation_b']
    image = Image.open(image_path)
    question = conversation_a[0]['content']
    return image, question, model_a, conversation_a[1]['content'], model_b, conversation_b[1]['content']

def build_side_by_side_ui_failure_case():
    # with gr.Blocks() as demo:
    gr.Markdown(HEADER_MD)
    gr.Markdown('Failure Cases Examples, Click to shuffle!')
    # user_image = gr.Gallery(label="Input images", show_label=True, elem_id="user_image", columns=[1], rows=[1], object_fit="contain", height=550)
    # user_image = gr.Image(type="pil", height=550)
    with gr.Row():
        with gr.Column(scale=1):
            user_image = gr.Image(label="Input image", show_label=True, type='pil', height='auto')
            user_question = gr.Textbox(label="Question", max_lines=2, interactive=False)

        with gr.Column():
            model_a_name = gr.Textbox(label="Model A", max_lines=2, interactive=False)
            model_a_response = gr.Textbox(label="Model A Response", max_lines=10, interactive=False)
        with gr.Column():
            model_b_name = gr.Textbox(label="Model B", max_lines=2, interactive=False)
            model_b_response = gr.Textbox(label="Model B Response", max_lines=10, interactive=False)
    btn = gr.Button("Sample Failure Case", scale=0)
    btn.click(sample_case, inputs=None, outputs=[user_image, user_question, model_a_name, model_a_response, model_b_name, model_b_response])
        # demo.launch(share=True)


if __name__ == "__main__":
    build_side_by_side_ui_failure_case()