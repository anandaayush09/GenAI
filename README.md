# Video → Avatar Narration

FastAPI backend that turns a video into a version with AI-avatar introductions in front of its key moments.

**Pipeline:** upload → ffmpeg audio → faster-whisper transcript → Gemini cleanup → Google Sheet (you can edit it) → Gemini picks key segments and writes narration → HeyGen renders the avatar → ffmpeg stitches intro + original clip.

## Setup
Prereqs: Python 3.10+, [ffmpeg](https://ffmpeg.org/) (with `ffprobe`) on your PATH, a Gemini API key, a HeyGen API key, and a Google service-account JSON key with the Sheets and Drive APIs enabled.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then fill in the keys; put your service-account JSON next to main.py
```

## Run
```bash
python main.py                  # API on http://127.0.0.1:8000 (interactive docs at /docs)
python client.py my_video.mp4   # in a second terminal: upload + build the narrated video
```
Output lands in `Data/Final_videos/` (git-ignored). Set `HOST=0.0.0.0` only if you really want it reachable from the network: the API has no authentication and spends HeyGen credits.

## Endpoints
| Method | Path | What it does |
|---|---|---|
| POST | `/process-video` | Multipart `file`. Transcribes, refines, creates a read-only-link Google Sheet, returns `spreadsheetId`. |
| GET | `/fetch-segments?sheetId=` | Returns the sheet's transcript rows. |
| POST | `/create-matthew-berman-style-video` | JSON `{sheetId, query?, avatar_id?, voice_id?}`. Builds the final video (minutes; runs synchronously). |
| POST | `/create-highlight-video` | Legacy alias of the above. |

## Configuration
See `.env.example`. Never commit `.env` or any `*.json` key (they are in `.gitignore`).

## Project notes
`docs/agent/` holds the architecture/flow, decision log and session handover. `_archive/` (git-ignored, local only) holds the old `Morning_*.py` / `backend*.py` copies.
