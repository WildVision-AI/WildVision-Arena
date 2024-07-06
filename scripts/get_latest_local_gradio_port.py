import os 

LOGDIR = os.getenv("WILDVISION_ARENA_LOGDIR", "/home/yuchenl/vision-arena-logs/")

log_file = f"{LOGDIR}/gradio_web_server.log"

# keep reading the file and find the LAST time where it contains "Running on public URL:"
urls = []
with open(log_file, "r") as f:
    lines = f.readlines()
    for line in reversed(lines):
        if "http://0.0.0.0:" in line: 
            port = line[line.index("http://0.0.0.0") + len("http://0.0.0.0") + 1: ].strip()
            urls.append(port)
            break

latest_url = urls[0]
print(latest_url)
"""
```bash 
GRADIO_PORT=$(python scripts/get_latest_local_gradio_port.py)
ssh -p 443 -R0:localhost:${GRADIO_PORT} 93Vo7sbIPXk@a.pinggy.io
```
"""