import http.server
import socketserver
import os


PORT = 5006

# Ensure that 5006 is okay
os.system(f"fuser -n tcp -k {PORT}")

# Set the folder you want to serve here
FOLDER_TO_SERVE = os.getenv('WEB_IMG_FOLDER', '/home/yuchenl/http_img')

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FOLDER_TO_SERVE, **kwargs)


with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Hosting the folder {FOLDER_TO_SERVE} at the port {PORT}.")
    httpd.serve_forever()
