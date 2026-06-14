import os
import urllib.request

os.makedirs("/app/pretrained_models", exist_ok=True)

WEIGHTS_URL = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
DEST = "/app/pretrained_models/matanyone2.pth"

if not os.path.exists(DEST):
    print("Downloading MatAnyone2 weights (~300MB)...")
    urllib.request.urlretrieve(WEIGHTS_URL, DEST)
    print("Done.")
else:
    print("Weights already present, skipping download.")
