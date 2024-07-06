import torch
import transformers

import sys
sys.path.append('./')
from videollama2.conversation import conv_templates, SeparatorStyle
from videollama2.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from videollama2.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, KeywordsStoppingCriteria, process_video, process_image
from videollama2.model.builder import load_pretrained_model


def inference():
    # Video Inference
    paths = ['assets/cat_and_chicken.mp4']
    questions = ['What animals are in the video, what are they doing, and how does the video feel?']
    # Reply:
    # The video features a kitten and a baby chick playing together. The kitten is seen laying on the floor while the baby chick hops around. The two animals interact playfully with each other, and the video has a cute and heartwarming feel to it.
    modal_list = ['video']

    # Video Inference
    paths = ['assets/sora.mp4']
    questions = ['Please describe this video.']
    # Reply:
    # The video features a series of colorful kites flying in the sky. The kites are first seen flying over trees, and then they are shown flying in the sky. The kites come in various shapes and colors, including red, green, blue, and yellow. The video captures the kites soaring gracefully through the air, with some kites flying higher than others. The sky is clear and blue, and the trees below are lush and green. The kites are the main focus of the video, and their vibrant colors and intricate designs are highlighted against the backdrop of the sky and trees. Overall, the video showcases the beauty and artistry of kite-flying, and it is a delight to watch the kites dance and glide through the air.
    modal_list = ['video']

    # Image Inference
    paths = ['assets/sora.png']
    questions = ['What is the woman wearing, what is she doing, and how does the image feel?']
    # Reply:
    # The woman in the image is wearing a black coat and sunglasses, and she is walking down a rain-soaked city street. The image feels vibrant and lively, with the bright city lights reflecting off the wet pavement, creating a visually appealing atmosphere. The woman's presence adds a sense of style and confidence to the scene, as she navigates the bustling urban environment.
    modal_list = ['image']

    # 1. Initialize the model.
    model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B'
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)
    model = model.to('cuda:0')
    conv_mode = 'llama_2'

    # 2. Visual preprocess (load & transform image or video).
    if modal_list[0] == 'video':
        tensor = process_video(paths[0], processor, model.config.image_aspect_ratio).to(dtype=torch.float16, device='cuda', non_blocking=True)
        default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
        modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
    else:
        tensor = process_image(paths[0], processor, model.config.image_aspect_ratio)[0].to(dtype=torch.float16, device='cuda', non_blocking=True)
        default_mm_token = DEFAULT_MMODAL_TOKEN["IMAGE"]
        modal_token_index = MMODAL_TOKEN_INDEX["IMAGE"]
    tensor = [tensor]

    # 3. Text preprocess (tag process & generate prompt).
    question = default_mm_token + "\n" + questions[0]
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_token_index, return_tensors='pt').unsqueeze(0).to('cuda:0')

    # 4. Generate a response according to visual signals and prompts. 
    stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE] else conv.sep2
    # keywords = ["<s>", "</s>"]
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images_or_videos=tensor,
            modal_list=modal_list,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print(outputs[0])


if __name__ == "__main__":
    inference()
