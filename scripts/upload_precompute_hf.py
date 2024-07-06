import os
import fire
import datasets
from datasets import load_dataset
import itertools
from icecream import ic
        
from openai import OpenAI
import base64
from io import BytesIO

def create_hf_battle_dataset(data_file: str, split="test"):
    hf_dataset = datasets.Dataset.from_list(
        data_file,
        features=datasets.Features(
            {
                "question_id": datasets.Value("string"),
                "model_a": datasets.Value("string"),
                "model_b": datasets.Value("string"),
                "conversation_a": [
                    {
                        "role": datasets.Value("string"),
                        "content": datasets.Value("string"),
                    }
                ],
                "conversation_b": [
                    {
                        "role": datasets.Value("string"),
                        "content": datasets.Value("string"),
                    }
                ],
                "language": datasets.Value("string"),
                "image": datasets.Image(),
                "turn": datasets.Value("int32"),
                "anony": datasets.Value("bool"),
                "tstamp": datasets.Value("int32"),
                "judge": datasets.Value("string"),
                "winner": datasets.Value("string"),
            }
        ),
        split=split,
    )
    return hf_dataset

def load_model_gen_results():
    model_dataset = {}
    for model_name in ["gpt-4-vision-preview", "gemini-pro-vision", "Qwen/Qwen-VL-Chat", "Salesforce/instructblip-vicuna-7b", "openbmb/MiniCPM-V", "THUDM/cogvlm-chat-hf", "liuhaotian/llava-v1.6-vicuna-7b", "liuhaotian/llava-v1.6-34b"]:
        model_dataset[model_name] = datasets.load_from_disk(f'gen_results/{model_name}')
    return model_dataset

task_template_map = {
    "pair_rate_wexplanation": "[Instruction]\n\"{instruction}\"\n\n\"{generated_sentence}\"[System]\nPlease act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any positional biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.",
    "pair_rate": "[Instruction]\n\"{instruction}\"\n\n\"{generated_sentence}\"\n\n[System]\nPlease act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any positional biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Reply with \"leftvote\" if you find assistant A better, \"rightvote\" if assistant B is better, \"bothbad_vote\" if both responses are wrong, and \"tievote\" if both assistants provide equally satisfactory answers. If you are unable to make a decision, please reply with \"NA\"."
}

def convert_pil_to_base64(image):
    # convert pil image to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode('utf-8')

def get_gpt4v_response(client, img_bs64=None, text_prompt="", use_vision=False):
    if use_vision:
        response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_bs64}"
                        }
                    },
                ],
            }
        ],
        max_tokens=100,
        )
    else:
        response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                ],
            }
        ],
        max_tokens=100,
        )
    return response.choices[0].message.content

def main():
    dataset = load_dataset("WildVision/wildvision-bench", "release_100_as_bench", split="test")
    model_dataset = load_model_gen_results()
    token = os.environ.get("HUGGINGFACE_TOKEN", None),
    new_data = []
    battle_cnt = 0
    client = OpenAI()
    for battle in dataset:

        ic(battle["question_id"], battle_cnt)
        for model_a, model_b in itertools.combinations(model_dataset.keys(), 2):
            # get conversation from model_a that has the same question_id
            # conversation_a = [d["conversation"] for d in model_dataset[model_a] if d["question_id"] == battle["question_id"]]
            # conversation_b = [d["conversation"] for d in model_dataset[model_b] if d["question_id"] == battle["question_id"]]
            conversation_a = model_dataset[model_a][battle_cnt]["conversation"]
            conversation_b = model_dataset[model_b][battle_cnt]["conversation"]
            assert model_dataset[model_a][battle_cnt]["question_id"] == battle["question_id"]
            base64_image = convert_pil_to_base64(battle["image"])
            left_response = conversation_a[1]["content"]
            right_response = conversation_b[1]["content"]
            instruction = conversation_a[0]["content"]
            generated_sentence = f"[The Start of Assistant A’s Answer]\n{left_response}\n[The End of Assistant A’s Answer]\n\n[The Start of Assistant B’s Answer]\n{right_response}\n[The End of Assistant B’s Answer]"
            text_prompt = task_template_map["pair_rate"].format(instruction=instruction, generated_sentence=generated_sentence)
            # ic(text_prompt)
            try:
                response = get_gpt4v_response(client, img_bs64=base64_image, text_prompt=text_prompt, use_vision=True)
            except:
                ic(">>> skip")
                response = "NA"
            
            # response = get_gpt4v_response(client, img_bs64=base64_image, text_prompt=text_prompt, use_vision=True)
            if response.strip() not in ["leftvote", "rightvote", "bothbad_vote", "tievote"]:
                winner = "tie"
            elif response.strip() == "leftvote":
                winner = "model_a"
            elif response.strip() == "rightvote":
                winner = "model_b"
            elif response.strip() == "bothbad_vote":
                winner = "tie (bothbad)"
            elif response.strip() == "tievote":
                winner = "tie"
                
            new_data.append({
                "question_id": battle["question_id"],
                "model_a": model_a,
                "model_b": model_b,
                "conversation_a": conversation_a,
                "conversation_b": conversation_b,
                "language": battle["language"],
                "image": battle["image"],
                "turn": battle["turn"],
                "anony": True,
                "winner": winner, #TODO: get winner form gpt4v
                "tstamp": None,
                "judge": "gpt4v",
            })
        battle_cnt += 1
        # if battle_cnt > 20: break
        # new_data.append({
        #     "question_id": battle["question_id"],
        #     "model_a": battle["model_a"],
        #     "model_b": battle["model_b"],
        #     "conversation_a": battle["conversation_a"],
        #     "conversation_b": battle["conversation_b"],
        #     "language": battle["language"],
        #     "image": battle["image"],
        #     "turn": battle["turn"],
        #     "anony": battle["anony"],
        #     "winner": battle["winner"],
        #     "tstamp": battle["tstamp"],
        #     "judge": battle["judge"],
        # })
    split = "precompute_gpt4v_vote"
    hf_dataset = create_hf_battle_dataset(new_data, split)

    hf_dataset.push_to_hub(
        repo_id="WildVision/wildvision-bench",
        config_name="release_100_as_bench_battle",
        split=split,
        token=token,
        commit_message=f"Add vision-arena {split} dataset",
    )
    
    print("Done!")
    
if __name__ == "__main__":
    fire.Fire(main)