"""Call API providers."""

from json import loads
import os
import random
import time

from arena.utils import build_logger
from arena.constants import WORKER_API_TIMEOUT

import base64
from io import BytesIO

from icecream import ic

logger = build_logger("gradio_web_server", "gradio_web_server.log")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def convert_pil_to_base64(image):
    # convert pil image to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode('utf-8')
    
def generate(model_name, gen_params, image, messages, is_yivl_api=False):
    from openai import OpenAI
    img_bs64 = convert_pil_to_base64(image)
    if is_yivl_api:
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=os.getenv("YIVL_API_KEY"),
            base_url=os.getenv("YIVL_API_BASE")
        )
    else:
        client = OpenAI()
    input_messages = messages
    # FIXME: support various images in history; similar to what we did in reka messages
    if image is not None:
        # text_message = input_messages[1]["content"]
        text_message = input_messages[-1]["content"]
        # input_messages[1]["content"] = [
        input_messages[-1]["content"] = [
            {"type": "text", "text": text_message},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_bs64}"
                }
            }
        ]
    ic(model_name)
    response = client.chat.completions.create(
    model=model_name,#"yi-vl-plus" if is_yivl_api else "gpt-4-vision-preview",
    messages=input_messages,
    max_tokens=min(int(gen_params.get("max_new_tokens", 1024)), 1024),
    temperature=float(gen_params.get("temperature", 0.2)),
    top_p = float(gen_params.get("top_p", 0.7)),
    stream=True if is_yivl_api else False
    )
    if is_yivl_api:
        return response
    else:
        return response.choices[0].message.content
    
def openai_api_stream_iter(
    model_name,
    messages,
    temperature,
    top_p,
    max_new_tokens,
    image=None,
    api_base=None,
    api_key=None,
):
    import openai

    is_azure = False
    if "azure" in model_name:
        is_azure = True
        openai.api_type = "azure"
        openai.api_version = "2023-07-01-preview"
    else:
        openai.api_type = "open_ai"
        openai.api_version = None

    openai.api_base = api_base or "https://api.openai.com/v1"
    openai.api_key = api_key or os.environ["OPENAI_API_KEY"]
    # if model_name == "gpt-4-turbo":
    #     model_name = "gpt-4-1106-preview"

    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    logger.info(f"==== request ====\n{gen_params}")

    if image is not None:
        res = generate(model_name, gen_params, image, messages)
        data = {
            "text": res,
            "error_code": 0,
        }
        yield data
        # TODO: gpt4-vision api not supporting stream yet
        # text = ""
        # for chunk in res:
        #     if len(chunk.choices) > 0:
        #         text += chunk.choices[0].delta.content
        #         data = {
        #             "text": text,
        #             "error_code": 0,
        #         }
        #         yield data
    else:
        if is_azure:
            res = openai.ChatCompletion.create(
                engine=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
                stream=True,
            )
        else:
            res = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
                stream=True,
            )
        text = ""
        for chunk in res:
            if len(chunk["choices"]) > 0:
                text += chunk["choices"][0]["delta"].get("content", "")
                data = {
                    "text": text,
                    "error_code": 0,
                }
                yield data


def anthropic_api_stream_iter(model_name, prompt, temperature, top_p, max_new_tokens, image=None):
    import anthropic

    c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    
    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    logger.info(f"==== request ====\n{gen_params}")
    if image is not None:
        img_bs64 = convert_pil_to_base64(image)
        message = c.messages.create(
            model=model_name,
            max_tokens=max_new_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_bs64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )
        data = {
            "text": message.content[0].text,
            "error_code": 0,
        }
        yield data
    else:
        res = c.completions.create(
            prompt=prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            max_tokens_to_sample=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            model=model_name,
            stream=True,
        )
        text = ""
        for chunk in res:
            text += chunk.completion
            data = {
                "text": text,
                "error_code": 0,
            }
            yield data


def init_palm_chat(model_name):
    import vertexai  # pip3 install google-cloud-aiplatform
    from vertexai.preview.language_models import ChatModel
    from vertexai.preview.generative_models import GenerativeModel

    project_id = os.environ["GCP_PROJECT_ID"]
    location = "us-central1"
    vertexai.init(project=project_id, location=location)

    if model_name in ["palm-2"]:
        # According to release note, "chat-bison@001" is PaLM 2 for chat.
        # https://cloud.google.com/vertex-ai/docs/release-notes#May_10_2023
        model_name = "chat-bison@001"
        chat_model = ChatModel.from_pretrained(model_name)
        chat = chat_model.start_chat(examples=[])
    elif model_name in ["gemini-pro"]:
        model = GenerativeModel(model_name)
        chat = model.start_chat()
    return chat


def gemini_vision_api_stream_iter(model_name, message, temperature, top_p, max_new_tokens, image):
    import google.generativeai as genai

    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(model_name)

    # no streaming version
    response = model.generate_content([message, image])
    response.resolve()
    
    output_result = response.text
    data = {
        "text": output_result,
        "error_code": 0,
    }
    yield data

    # response = model.generate_content([message, image], stream=True)
    # generate_text = ""
    # for chunk in response:
    #     generate_text += chunk.text
    #     data = {
    #         "text": generate_text,
    #         "error_code": 0,
    #     }
    #     yield data
    
def https_image_service(image):
    from arena.constants import WEB_IMG_FOLDER
        
    WEB_IMG_URL_ROOT = os.getenv("WEB_IMG_URL_ROOT")
    import shortuuid
    img_id = shortuuid.uuid()
    
    image.save(os.path.join(WEB_IMG_FOLDER, f"{img_id}.png"))
    media_url = f"{WEB_IMG_URL_ROOT}/{img_id}.png"
    return media_url
    
def idefics2_api_stream_iter(model_name, conv, temperature, top_p, max_new_tokens, image):
    from text_generation import Client
    import google.generativeai as genai
    HF_API_TOKEN = os.getenv('HF_API_TOKEN')
    API_URL = "https://api-inference.huggingface.co/models/HuggingFaceM4/idefics2-8b-chatty"
    
    SYSTEM_PROMPT = "System: The following is a conversation between Idefics2, a highly knowledgeable and intelligent visual AI assistant created by Hugging Face, referred to as Assistant, and a human user called User. In the following interactions, User and Assistant will converse in natural language, and Assistant will do its best to answer User’s questions. Assistant has the ability to perceive images and reason about them, but it cannot generate images. Assistant was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. When prompted with an image, it does not make up facts.<end_of_utterance>\nAssistant: Hello, I'm Idefics2, Huggingface's latest multimodal assistant. How can I help you?<end_of_utterance>\n"
    media_url = https_image_service(image) # dev: "https://raw.githubusercontent.com/huggingface/text-generation-inference/main/integration-tests/images/chicken_on_money.png"
    conv.set_media_url(media_url)
    prompt = conv.to_idefics2_messages()
    # ic(conv, prompt)
    QUERY = prompt #f"User:![]({media_url})Describe this image.<end_of_utterance>\nAssistant:"

    client = Client(
        base_url=API_URL,
        headers={"x-use-cache": "0", "Authorization": f"Bearer {HF_API_TOKEN}"},
    )
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": 1.1,
        "temperature": max(min(temperature, 1.0-1e-3), 1e-3),
        # "top_p": max(min(top_p, 1.0-1e-3), 1e-3),
        "do_sample": True if temperature > 1e-3 else False,
    }
    # generated_text = client.generate(prompt=SYSTEM_PROMPT + QUERY, **generation_args)
    # data = {
    #     "text": generated_text,
    #     "error_code": 0,
    # }
    from icecream import ic
    response = client.generate_stream(prompt=SYSTEM_PROMPT + QUERY, **generation_args)
    # ic(response.token.text)
    text = ""
    for response in client.generate_stream(prompt=SYSTEM_PROMPT + QUERY, **generation_args):
        if not response.token.special:
            text += response.token.text
            data = {
                "text": text,
                "error_code": 0,
            }
            yield data

def minicpm_api_stream_iter(model_name, conv, temperature, top_p, max_new_tokens, image):
    import base64
    import hashlib
    import hmac
    import json
    import uuid
    import requests
    from datetime import datetime#, timedeltas
    class OpenapiClient:
        def __init__(self, app_id: str, app_key: str):
            self.app_id = app_id
            self.app_key = app_key

        @staticmethod
        def get_md5_base64(s: str) -> str:
            if str is None:
                return None
            md5 = hashlib.md5(s.encode("utf-8")).digest()
            return base64.b64encode(md5).decode("utf-8")

        @staticmethod
        def get_signature_base64(data: str, key: str) -> str:
            key_bytes = key.encode("utf-8")
            data_bytes = data.encode("utf-8")
            hmac_obj = hmac.new(key_bytes, data_bytes, hashlib.sha256)
            return base64.b64encode(hmac_obj.digest()).decode("utf-8")


        def do_http_post(self, msgs) -> None:
            try:
                url_queries = None
                json_data = {
                    "model": "MiniCPM-Llama3-V-2.5",
                    "messages": msgs
                }
                content_md5 = OpenapiClient.get_md5_base64(json.dumps(json_data))
                method = "POST"
                accept = "*/*"
                content_type = "application/json"
                # timestamp = int((datetime.utcnow() + timedelta(hours=8)).timestamp() * 1000)
                from datetime import timezone
                timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
                mode = "Signature"
                nonce = str(uuid.uuid4())

                sbuffer = "\n".join(
                    [
                        method,
                        accept,
                        content_type,
                        str(timestamp),
                        content_md5,
                        mode,
                        nonce,
                        url_queries if url_queries else "",
                    ]
                )
                signature = OpenapiClient.get_signature_base64(sbuffer, self.app_key)

                headers = {
                    "Content-Type": content_type,
                    "Accept": accept,
                    "X-Model-Best-Open-Ca-Time": str(timestamp),
                    "Content-MD5": content_md5,
                    "X-Model-Best-Open-App-Id": self.app_id,
                    "X-Model-Best-Open-Ca-Mode": mode,
                    "X-Model-Best-Open-Ca-Nonce": nonce,
                    "X-Model-Best-Open-Ca-Signature": signature,
                }

                api_url = "https://openapi.modelbest.cn/openapi/v2/chat/completions"
                response = requests.post(api_url, headers=headers, json=json_data)
                return response.json()
            except Exception as e:
                # TODO: log error
                raise e
    app_id = "4a011e5e-2c4b-4edb-90d7-86fcab863159"
    app_key = "7y2Njl2-m1Dpb3AhFbUvbG_9zwA1gKWe"
    cli = OpenapiClient(app_id, app_key)
    encoded_image = convert_pil_to_base64(image)
    # conv.set_encoded_image(encoded_image)
    conv.set_encoded_image(encoded_image)
    msgs= conv.to_minicpm_messages()
    # ic(msgs)
    res = cli.do_http_post(msgs)
    # res = {'id': 'AG-7ogtUfiss7VydmN91o', 'choices': [{'index': None, 'message': {'role': 'assistant', 'content': 'The image contains a mathematical derivation related to reinforcement learning algorithms, specifically the DPO (Dynamic Programming Optimization) objective and its application in policy optimization.', 'contents': None, 'tool_calls': []}, 'finish_reason': 'MESSAGE'}], 'created': 1717094032380, 'model': 'MiniCPM-Llama3-V-2.5', 'object': None, 'usage': {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}, 'status': 'success', 'system_fingerprint': None}
    ic(res)
    res = res['choices'][0]['message']
    
    output_result = res['content']
    data = {
        "text": output_result,
        "error_code": 0,
    }
    yield data

def yivl_api_stream_iter(model_name, messages, temperature, top_p, max_new_tokens, image):
    # from arena.constants import WEB_IMG_FOLDER
    # client = OpenAI(
    #     # defaults to os.environ.get("OPENAI_API_KEY")
    #     api_key=os.getenv("YIVL_API_KEY"),
    #     base_url=os.getenv("YIVL_API_BASE")
    # )
    # WEB_IMG_URL_ROOT = os.getenv("WEB_IMG_URL_ROOT")
    # import shortuuid
    # img_id = shortuuid.uuid()
    
    # image.save(os.path.join(WEB_IMG_FOLDER, f"{img_id}.png"))
    # completion = client.chat.completions.create(
    #     model="yi-vl-plus",
    #     messages=[{"role": "user",
    #                 "content": [
    #                     {
    #                         "type": "image_url",
    #                         "image_url": {
    #                             "url": f"{WEB_IMG_URL_ROOT}/{img_id}.png"
    #                         }
    #                     },
    #                     {
    #                         "type": "text",
    #                         "text": prompt[0]["content"]
    #                     }
    #                 ]
    #             }],
    #     stream=True
    # )
    gen_params = {
        "model": model_name,
        "prompt": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    completion = generate(model_name, gen_params, image, messages, is_yivl_api=True)
                
    text = ""
    for chunk in completion:
        text += chunk.choices[0].delta.content or ""
        # print(chunk.choices[0].delta.content or "", end="", flush=True)
        data = {
            "text": text,
            "error_code": 0,
        }
        yield data
        
def llava_api_stream_iter(model_name, messages, temperature, top_p, max_new_tokens, image):
    gen_params = {
        "prompt": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    print(gen_params)
    img_bs64 = convert_pil_to_base64(image)
    client = OpenAI(
        api_key=os.getenv("LLAVA_API_KEY"),
        base_url=os.getenv("LLAVA_API_BASE")
    )
    input_messages = messages
    if image is not None:
        text_message = input_messages[0]["content"]
        input_messages[0]["content"] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_bs64}"
                }
            },
            {"type": "text", "text": text_message},
        ]
    ic(model_name)
    response = client.chat.completions.create(
        model="default",
        messages=input_messages,
        max_tokens=min(int(gen_params.get("max_new_tokens", 1024)), 1024),
        temperature=float(gen_params.get("temperature", 0.2)),
        top_p = float(gen_params.get("top_p", 0.7)),
        stream=True
    )
                
    text = ""
    for chunk in response:
        text += chunk.choices[0].delta.content or ""
        # print(chunk.choices[0].delta.content or "", end="", flush=True)
        data = {
            "text": text,
            "error_code": 0,
        }
        yield data

def reka_api_stream_iter(model_name, conv, temperature, top_p, max_new_tokens, image):
    import reka

    media_url = https_image_service(image)

    conv.set_media_url(media_url)
    conv.set_media_type("image")
    prompt = conv.to_reka_api_messages()

    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    if len(prompt) <= 2:
        response = reka.chat(
            prompt[-2]["text"],
            media_url=media_url,
            media_type="image",
            request_output_len=gen_params["max_new_tokens"],
            temperature=gen_params["temperature"],
            # runtime_top_k=1024,
            runtime_top_p=gen_params["top_p"],
        )
    else:
        # support mult-turn Reka
        response = reka.chat(
            prompt[-2]["text"],
            conversation_history=prompt[:-2],
            request_output_len=gen_params["max_new_tokens"],
            temperature=gen_params["temperature"],
            # runtime_top_k=1024,
            runtime_top_p=gen_params["top_p"],
        )

    data = {
        "text": response["text"],
        "error_code": 0,
    }
    yield data
    
def reka_api_stream_iter(model_name, conv, temperature, top_p, max_new_tokens, image):
    import reka

    media_url = https_image_service(image)

    conv.set_media_url(media_url)
    conv.set_media_type("image")
    prompt = conv.to_reka_api_messages()

    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    if len(prompt) <= 2:
        response = reka.chat(
            prompt[-2]["text"],
            media_url=media_url,
            media_type="image",
            request_output_len=gen_params["max_new_tokens"],
            temperature=gen_params["temperature"],
            # runtime_top_k=1024,
            runtime_top_p=gen_params["top_p"],
        )
    else:
        # support mult-turn Reka
        response = reka.chat(
            prompt[-2]["text"],
            conversation_history=prompt[:-2],
            request_output_len=gen_params["max_new_tokens"],
            temperature=gen_params["temperature"],
            # runtime_top_k=1024,
            runtime_top_p=gen_params["top_p"],
        )

    data = {
        "text": response["text"],
        "error_code": 0,
    }
    yield data

def qwenvl_api_stream_iter(model_name, conv, temperature, top_p, max_new_tokens, image):
    from http import HTTPStatus
    import dashscope

    # gen_params = {
    #     "model": model_name,
    #     "prompt": prompt,
    #     "temperature": temperature,
    #     "top_p": top_p,
    #     "max_new_tokens": max_new_tokens,
    # }
    
    media_url = https_image_service(image)

    conv.set_media_url(media_url)
    conv.set_media_type("image")
    messages = conv.to_qwenvlapi_messages()
    ic(messages)
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"image": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"},
    #             {"text": "这是什么?"}
    #         ]
    #     }
    # ]

    response = dashscope.MultiModalConversation.call(model=model_name,
                                                     messages=messages)
    # ic(response)
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    # if response.status_code == HTTPStatus.OK:
    #     print(response)
    # else:
    #     print(response.code)  # The error code.
    #     print(response.message)  # The error message.

    data = {
        "text": response.output.choices[0].message.content[0]["text"],
        "error_code": 0,
    }
    yield data


def palm_api_stream_iter(model_name, chat, message, temperature, top_p, max_new_tokens):
    if model_name in ["gemini-pro"]:
        max_new_tokens = max_new_tokens * 2
    parameters = {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_new_tokens,
    }
    gen_params = {
        "model": model_name,
        "prompt": message,
    }
    gen_params.update(parameters)
    if model_name == "palm-2":
        response = chat.send_message(message, **parameters)
    else:
        response = chat.send_message(message, generation_config=parameters, stream=True)

    logger.info(f"==== request ====\n{gen_params}")

    try:
        text = ""
        for chunk in response:
            text += chunk.text
            data = {
                "text": text,
                "error_code": 0,
            }
            yield data
    except Exception as e:
        logger.error(f"==== error ====\n{e}")
        yield {
            "text": f"**API REQUEST ERROR** Reason: {e}\nPlease try again or increase the number of max tokens.",
            "error_code": 1,
        }
        yield data


def ai2_api_stream_iter(
    model_name,
    messages,
    temperature,
    top_p,
    max_new_tokens,
    api_key=None,
    api_base=None,
):
    from requests import post

    # get keys and needed values
    ai2_key = api_key or os.environ.get("AI2_API_KEY")
    api_base = api_base or "https://inferd.allen.ai/api/v1/infer"
    model_id = "mod_01hhgcga70c91402r9ssyxekan"

    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    logger.info(f"==== request ====\n{gen_params}")

    # AI2 uses vLLM, which requires that `top_p` be 1.0 for greedy sampling:
    # https://github.com/vllm-project/vllm/blob/v0.1.7/vllm/sampling_params.py#L156-L157
    if temperature == 0.0 and top_p < 1.0:
        raise ValueError("top_p must be 1 when temperature is 0.0")

    res = post(
        api_base,
        stream=True,
        headers={"Authorization": f"Bearer {ai2_key}"},
        json={
            "model_id": model_id,
            # This input format is specific to the Tulu2 model. Other models
            # may require different input formats. See the model's schema
            # documentation on InferD for more information.
            "input": {
                "messages": messages,
                "opts": {
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "logprobs": 1,  # increase for more choices
                },
            },
        },
    )

    if res.status_code != 200:
        logger.error(f"unexpected response ({res.status_code}): {res.text}")
        raise ValueError("unexpected response from InferD", res)

    text = ""
    for line in res.iter_lines():
        if line:
            part = loads(line)
            if "result" in part and "output" in part["result"]:
                for t in part["result"]["output"]["text"]:
                    text += t
            else:
                logger.error(f"unexpected part: {part}")
                raise ValueError("empty result in InferD response")

            data = {
                "text": text,
                "error_code": 0,
            }
            yield data
