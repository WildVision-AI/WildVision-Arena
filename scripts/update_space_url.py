import os 
from huggingface_hub import HfApi
from datetime import datetime
from urllib.parse import urlparse
from requests import get

LOGDIR = os.getenv("WILDVISION_ARENA_LOGDIR", "/home/yuchenl/vision-arena-logs/")

log_file = f"{LOGDIR}/gradio_web_server.log"

# keep reading the file and find the LAST time where it contains "Running on public URL:"
urls = []
with open(log_file, "r") as f:
    lines = f.readlines()
    for line in reversed(lines):
        if "Running on public URL:" in line:
            assert "https:" in line and ".gradio.live" in line
            url = line[line.index("https:") : line.index(".gradio.live") + len(".gradio.live")]
            urls.append(url.strip())
            break

latest_url = urls[0]

# check if latest_url is in "https://huggingface.co/spaces/WildVision/vision-arena/raw/main/index.html" content

current_html = get("https://huggingface.co/spaces/WildVision/vision-arena/raw/main/index.html").text
if latest_url in current_html:
    print(f"{datetime.now()} - The latest URL is already in the index.html")
    exit(0)


api = HfApi() 

html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vision Arena by WildVision Team</title>
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            var gradioURL = "{latest_url}/?__theme=light"; // Your variable URL
            var iframe = document.getElementById("gradioIframe");
            var link = document.getElementById("gradioLink");
            if (iframe) iframe.src = gradioURL;
            if (link) link.href = gradioURL;
        });
    </script>
</head>
<body>
    <iframe id="gradioIframe" width="100%" height="100%" style="border:none;">
        Your browser does not support iframes. Please click this <a id="gradioLink">url</a>. 
    </iframe>
</body>
</html>
'''.replace("{latest_url}", latest_url)

api.upload_file(
        repo_id="WildVision/vision-arena",
        repo_type="space",
        path_in_repo="index.html",
        # Quick and dirty way to add a task
        path_or_fileobj=(html).encode()
    )


print(f"{datetime.now()} - Uploaded the latest URL to the space: {latest_url}")