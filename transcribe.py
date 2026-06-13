import json, os
from faster_whisper import WhisperModel

AUDIO_DIR = "audio"
TRANSCRIPT_DIR = "transcripts"
MODEL_SIZE = "small"

os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

with open('clips_metadata.json') as f:
    clips = json.load(f)

print(f'Found {len(clips)} clips')
print('Loading Whisper model...')
model = WhisperModel(MODEL_SIZE, device='cpu', compute_type='int8')
print('Model loaded\n')

results = []
for clip in clips:
    clip_id = clip['id']
    out_json = os.path.join(TRANSCRIPT_DIR, f"{clip_id}.json")
    out_txt  = os.path.join(TRANSCRIPT_DIR, f"{clip_id}.txt")

    if os.path.exists(out_json):
        print(f'[SKIP] {clip_id}')
        with open(out_json) as f:
            t = json.load(f)
    else:
        print(f'[TRANSCRIBE] {clip_id}: {clip["title"]}')
        segments, info = model.transcribe(clip['filepath'], beam_size=5, language='en')
        segs = [{'start': round(s.start,2), 'end': round(s.end,2), 'text': s.text.strip()} for s in segments]
        full = ' '.join(s['text'] for s in segs)
        t = {'language': info.language, 'duration_seconds': round(info.duration,1), 'segments': segs, 'full_text': full}
        with open(out_json, 'w') as f:
            json.dump(t, f, indent=2)
        with open(out_txt, 'w') as f:
            f.write(full)
        print(f'  -> {len(full.split())} words, {t["duration_seconds"]}s')

    results.append({**clip, 'transcript_json': out_json, 'transcript_txt': out_txt,
                    'word_count': len(t['full_text'].split()), 'duration_seconds': t.get('duration_seconds')})

with open('transcript_index.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nDone. {len(results)} transcripts saved to ./{TRANSCRIPT_DIR}/')
print('Next: run python index.py')