"""
The gradio demo server with multiple tabs.
It supports chatting with a single model or chatting with two models side-by-side.
"""

import argparse
import pickle
import time

import gradio as gr

from arena.constants import (
    SESSION_EXPIRATION_TIME,
)
from arena.serve.gradio_block_arena_anony_bench import (
    build_side_by_side_ui_anony_bench,
    load_demo_side_by_side_anony_bench,
    set_global_vars_anony_bench,
)
from arena.serve.gradio_block_arena_anony import (
    build_side_by_side_ui_anony,
    load_demo_side_by_side_anony,
    set_global_vars_anony,
)
from arena.serve.gradio_block_arena_anony_video import (
    build_side_by_side_ui_anony_video,
    load_demo_side_by_side_anony_video,
    set_global_vars_anony_video,
)
from arena.serve.gradio_block_arena_named import (
    build_side_by_side_ui_named,
    load_demo_side_by_side_named,
    set_global_vars_named,
)
from arena.serve.gradio_web_server import (
    set_global_vars,
    block_css,
    build_single_model_ui,
    build_single_model_chatbot,
    build_about,
    get_model_list,
    load_demo_single,
    ip_expiration_dict,
    get_ip,
)
from arena.serve.monitor.monitor import build_leaderboard_tab
from arena.utils import (
    build_logger,
    get_window_url_params_js,
    get_window_url_params_with_tos_js,
    parse_gradio_auth_creds,
)
from arena.serve.gradio_block_arena_show_failures import build_side_by_side_ui_failure_case
logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

from rich.console import Console
from rich.traceback import install

install()

console = Console()

def load_demo(url_params, request: gr.Request):
    global models

    ip = get_ip(request)
    logger.info(f"load_demo. ip: {ip}. params: {url_params}")
    ip_expiration_dict[ip] = time.time() + SESSION_EXPIRATION_TIME
    from icecream import ic
    ic(url_params)
    selected = 0
    if url_params is not None:
        if "arena" in url_params:
            selected = 0
        elif "compare" in url_params:
            selected = 1
        # elif "bench" in url_params:
        #     selected = 2
        elif "single" in url_params:
            selected = 2
        elif "leaderboard" in url_params:
            selected = 3

    if args.model_list_mode == "reload":
        if args.anony_only_for_proprietary_model:
            models = get_model_list(
                args.controller_url,
                args.register_openai_compatible_models,
                False,
                False,
                False,
            )
            video_model_list = get_model_list(
                args.controller_url,
                args.register_openai_compatible_models,
                False,
                False,
                False,
                model_type="video",
            )
            image_model_list = get_model_list(
                args.controller_url,
                args.register_openai_compatible_models,
                False,
                False,
                False,
                model_type="image",
            )
        else:
            models = get_model_list(
                args.controller_url,
                args.register_openai_compatible_models,
                args.add_chatgpt,
                args.add_claude,
                args.add_palm,
            )
            video_model_list = get_model_list(
                args.controller_url,
                args.register_openai_compatible_models,
                args.add_chatgpt,
                args.add_claude,
                args.add_palm,
                model_type="video",
            )
            image_model_list = get_model_list(
                args.controller_url,
                args.register_openai_compatible_models,
                args.add_chatgpt,
                args.add_claude,
                args.add_palm,
                model_type="image",
            )

    single_updates = load_demo_single(models, url_params)

    added_models_anony = []
    if args.anony_only_for_proprietary_model:
        # Only enable these models in anony battles.
        if args.add_chatgpt:
            added_models_anony += [
                "gpt-4-0314",
                "gpt-4-0613",
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-1106",
            ]
        if args.add_claude:
            added_models_anony += ["claude-2.1", "claude-2.0", "claude-1", "claude-instant-1"]
        if args.add_palm:
            added_models_anony += ["gemini-pro"]
    
    # TODO: uncomment to add LLM anony models
    # anony_only_models = [
    #     "claude-1",
    #     "gpt-4-0314",
    #     "gpt-4-0613",
    # ]
    # for mdl in anony_only_models:
    #     models_anony.append(mdl)
    # models_anony = list(set(models)) + added_models_anony # commented for it's not used
    image_model_list = list(set(image_model_list)) + added_models_anony
    # video_model_list = list(set(video_model_list)) + added_models_anony # commented for it's not used
    
    side_by_side_anony_updates = load_demo_side_by_side_anony(image_model_list, url_params)
    side_by_side_anony_video_updates = load_demo_side_by_side_anony_video(video_model_list, url_params)
    side_by_side_anony_bench_updates = load_demo_side_by_side_anony_bench(image_model_list, url_params)
    side_by_side_named_updates = load_demo_side_by_side_named(image_model_list, url_params)
    return (
        (gr.Tabs(selected=selected),)
        + single_updates
        + side_by_side_anony_updates
        + side_by_side_anony_video_updates
        + side_by_side_anony_bench_updates
        + side_by_side_named_updates
    )


def build_demo(models, elo_results_file, leaderboard_table_file, video_elo_results_file, video_leaderboard_table_file, show_sbs_direct=True):
    image_models = models["image"]
    video_models = models["video"]
    models = models["all"]
    text_size = gr.themes.sizes.text_md
    if args.show_terms_of_use:
        load_js = get_window_url_params_with_tos_js
    else:
        load_js = get_window_url_params_js
    with gr.Blocks(
        title="Chat with Open Multimodal Large Language Models",
        # theme=gr.themes.Default(text_size=text_size),
        theme=gr.themes.Soft(text_size=text_size),
        css=block_css,
        js=load_js
    ) as demo:
        with gr.Tabs() as tabs:
            with gr.Tab("⚔️ Arena ", elem_id="arena-tab", id=0):
                side_by_side_anony_list = build_side_by_side_ui_anony(image_models)
 
            with gr.Tab("⚔️ Video Arena ", elem_id="arena-tab", id=1):
                side_by_side_anony_video_list = build_side_by_side_ui_anony_video(video_models)

            if show_sbs_direct:               
                with gr.Tab("⚔️ Arena (side-by-side)", elem_id="arena-tab", id=2):
                    side_by_side_named_list = build_side_by_side_ui_named(image_models)

                # with gr.Tab("⚔️ Arena (bench)", id=2):
                #     side_by_side_anony_list = build_side_by_side_ui_anony_bench(models)

                # with gr.Tab("💬 Direct Chat", elem_id="arena-tab", id=2):
                #     single_model_list = build_single_model_ui(
                #         models, add_promotion_links=True
                #     )
                with gr.Tab("💬 Direct Chat", elem_id="arena-tab", id=3):
                    single_model_list = build_single_model_chatbot(
                        models, add_promotion_links=True
                    )
            else:
                single_model_list = []
                side_by_side_named_list = []


            if elo_results_file:
                with gr.Tab("🏆 Leaderboard", elem_id="arena-tab", id=4):
                    build_leaderboard_tab(elo_results_file, leaderboard_table_file, video_elo_results_file, video_leaderboard_table_file)
            with gr.Tab("ℹ️ About Us", elem_id="arena-tab", id=5):
                about = build_about()
            # with gr.Tab("👀 Failure Case Examples", elem_id="arena-tab", id=6):
            #     failure_cases = build_side_by_side_ui_failure_case()

        url_params = gr.JSON(visible=False)

        if args.model_list_mode not in ["once", "reload"]:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")


        demo.load(
            load_demo,
            [url_params],
            [tabs]
            + single_model_list
            + side_by_side_anony_list
            + side_by_side_anony_video_list
            + side_by_side_named_list,
            # _js=load_js,
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
        default="reload",
        choices=["once", "reload"],
        help="Whether to load the model list once or reload the model list every time.",
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
        default=False,
        help="Add OpenAI's ChatGPT models (gpt-3.5-turbo, gpt-4)",
    )
    parser.add_argument(
        "--add-claude",
        action="store_true",
        default=False,
        help="Add Anthropic's Claude models (claude-2, claude-instant-1)",
    )
    parser.add_argument(
        "--add-palm",
        action="store_true",
        default=False,
        help="Add Google's PaLM model (PaLM 2 for Chat: chat-bison@001)",
    )
    parser.add_argument(
        "--show-error-code",
        action="store_true",
        help="Show deveolopment error code in the response",
    )
    parser.add_argument(
        "--anony-only-for-proprietary-model",
        action="store_true",
        default=False,
        help="Only add ChatGPT, Claude, Bard under anony battle tab",
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
        default=None,
    )
    parser.add_argument(
        "--elo-results-file", type=str, help="Load leaderboard results and plots"
    )
    parser.add_argument(
        "--leaderboard-table-file", type=str, help="Load leaderboard results and plots"
    )
    parser.add_argument(
        "--video-elo-results-file", type=str, help="Load leaderboard results and plots for video models"
    )
    parser.add_argument(
        "--video-leaderboard-table-file", type=str, help="Load leaderboard results and plots for video models"
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
    set_global_vars_named(args.moderate)
    set_global_vars_anony(args.moderate)
    set_global_vars_anony_video(args.moderate)
    set_global_vars_anony_bench(args.moderate)
    if args.anony_only_for_proprietary_model:
        models = get_model_list(
            args.controller_url,
            args.register_openai_compatible_models,
            False,
            False,
            False,
        )
        video_model_list = get_model_list(
            args.controller_url,
            args.register_openai_compatible_models,
            False,
            False,
            False,
            model_type="video",
        )
        image_model_list = get_model_list(
            args.controller_url,
            args.register_openai_compatible_models,
            False,
            False,
            False,
            model_type="image",
        )
    else:
        models = get_model_list(
            args.controller_url,
            args.register_openai_compatible_models,
            args.add_chatgpt,
            args.add_claude,
            args.add_palm,
        )
        video_model_list = get_model_list(
            args.controller_url,
            args.register_openai_compatible_models,
            args.add_chatgpt,
            args.add_claude,
            args.add_palm,
            model_type="video",
        )
        image_model_list = get_model_list(
            args.controller_url,
            args.register_openai_compatible_models,
            args.add_chatgpt,
            args.add_claude,
            args.add_palm,
            model_type="image",
        )
    
    all_models = {
        "all": models,
        "image": image_model_list,
        "video": video_model_list
    }

    # Set authorization credentials
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)

    # Launch the demo
    demo = build_demo(all_models, args.elo_results_file, args.leaderboard_table_file, args.video_elo_results_file, args.video_leaderboard_table_file)
    # deprecation: concurrency_count=args.concurrency_count, 
    demo.queue(
        status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
        auth=auth,
        auth_message="Your conversation will be logged for research purposes. Please do not share personal information.",
        root_path=args.gradio_root_path
    )
