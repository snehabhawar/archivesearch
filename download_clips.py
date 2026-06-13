"""
download_clips.py — fixed URLs verified June 2026
All audio is public domain (LibriVox CC0 recordings of FDR Fireside Chats).
"""

import os, json, requests
from tqdm import tqdm

CLIPS = [
    {"id": "arch_001", "title": "FDR Fireside Chat 01 - Banking Crisis 1933",
     "source": "https://archive.org/download/firesidechats_1705_librivox/firesidechats_01_roosevelt.mp3",
     "year": 1933, "topic_hint": "economy"},
    {"id": "arch_002", "title": "FDR Fireside Chat 02 - Recovery Programs 1934",
     "source": "https://archive.org/download/firesidechats_1705_librivox/firesidechats_02_roosevelt.mp3",
     "year": 1934, "topic_hint": "economy"},
    {"id": "arch_003", "title": "FDR Fireside Chat 03 - New Deal Progress 1934",
     "source": "https://archive.org/download/firesidechats_1705_librivox/firesidechats_03_roosevelt.mp3",
     "year": 1934, "topic_hint": "politics"},
    {"id": "arch_004", "title": "FDR Fireside Chat 04 - Drought and Economics 1934",
     "source": "https://archive.org/download/firesidechats_1705_librivox/firesidechats_04_roosevelt.mp3",
     "year": 1934, "topic_hint": "economy"},
    {"id": "arch_005", "title": "FDR Fireside Chat 05 - Works Relief Program 1935",
     "source": "https://archive.org/download/firesidechats_1705_librivox/firesidechats_05_roosevelt.mp3",
     "year": 1935, "topic_hint": "government"},
    {"id": "arch_006", "title": "FDR Fireside Chat 06 - Social Security 1935",
     "source": "https://archive.org/download/firesidechats_1705_librivox/firesidechats_06_roosevelt.mp3",
     "year": 1935, "topic_hint": "society"},
    {"id": "arch_007", "title": "FDR Fireside Chat 07 - Tax Program 1935",
     "source": "https://archive.org/download/firesidechats_1705_librivox/firesidechats_07_roosevelt.mp3",
     "year": 1935, "topic_hint": "economy"},
    {"id": "arch_008", "title": "FDR Fireside Chat 08 - Drought Conditions 1936",
     "source": "https://archive.org/download/firesidechats_1705_librivox/firesidechats_08_roosevelt.mp3",
     "year": 1936, "topic_hint": "environment"},
    {"id": "arch_009", "title": "FDR Fireside Chat 09 - Judiciary Reorganization 1937",
     "source": "https://archive.org/download/firesidechats_1705_librivox/firesidechats_09_roosevelt.mp3",
     "year": 1937, "topic_hint": "government"},
    {"id": "arch_010", "title": "FDR Fireside Chat 10 - The Recession 1937",
     "source": "https://archive.org/download/firesidechats_1705_librivox/firesidechats_10_roosevelt.mp3",
     "year": 1937, "topic_hint": "economy"},
]

AUDIO_DIR = "audio"

def download(url, dest):
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                desc=os.path.basename(dest), leave=False) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [WARN] {url}: {e}")
        return False

def main():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    metadata = []
    print(f"Downloading {len(CLIPS)} public-domain audio clips...")
    for clip in CLIPS:
        filename = f"{clip['id']}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
            print(f"  [SKIP] {clip['id']} already exists")
        else:
            print(f"  [DOWN] {clip['id']}: {clip['title']}")
            if not download(clip["source"], filepath):
                continue
        metadata.append({**clip, "filename": filename, "filepath": filepath})
    with open("clips_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nDone. {len(metadata)} clips ready.")

if __name__ == "__main__":
    main()