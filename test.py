with open("examples/bigbang.mp4", 'rb') as video_file:
    video_bytes = video_file.read()
print(type(video_bytes))  # <class 'bytes'>
if type(video_bytes) == bytes:
    print("It is bytes")