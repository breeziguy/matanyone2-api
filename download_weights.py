import os
import urllib.request

os.makedirs("/app/weights", exist_ok=True)

WEIGHTS = {
    "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}

for filename, url in WEIGHTS.items():
    dest = f"/app/weights/{filename}"
    if not os.path.exists(dest):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, dest)
        print("Done.")
    else:
        print(f"{filename} already exists, skipping.")
