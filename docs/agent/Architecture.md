# Architecture

Single-module FastAPI app: `main.py` (server) + `client.py` (CLI driver). No database; state lives in files under `Data/` and in a Google Sheet per video.

## Request flow
`POST /process-video` (`process_video`)
→ save upload to `Data/Original_videos/<name>_<uuid>.mp4`
→ `extract_audio_with_ffmpeg` → `Data/Extracted_Audio/*.wav`
→ `get_whisper_model()` (cached faster-whisper "base", CPU int8) → segments
→ `call_gemini` refines text (fillers removed, timestamps kept) → rows
→ Sheets API creates a sheet titled with the video filename; Drive API shares it as **reader** link → `{spreadsheetId}`

`POST /create-matthew-berman-style-video` (`create_matthew_berman_style_video`)
→ read sheet rows A2:G (col B/C start/end, col G refined text)
→ `SegmentAnalyzer.analyze_segments_for_avatar_commentary` (Gemini → JSON list, top 6 by importance; deterministic fallback if the JSON is unusable)
→ find the original video in `Data/Original_videos/` **by sheet title** (this is the only link between sheet and video)
→ `AvatarVideoComposer`: per segment `HeyGenAvatarGenerator.generate_avatar_video` (POST, poll every 5 s up to 10 min, download) → ffmpeg cut of the original clip (min 10 s) → `_compose_final_video` (avatar+clip concat, then concat all) → `Data/Final_videos/ai_dynamic_<name>.mp4`

## Components
| Piece | Role |
|---|---|
| Gemini (`GEMINI_MODEL`, default `gemini-2.5-flash`) | transcript cleanup, segment choice, narration scripts |
| faster-whisper | local speech-to-text |
| HeyGen v2 API | avatar video from script text (forced 640×360) |
| Google Sheets/Drive (service account) | editable transcript store |
| ffmpeg / ffprobe | audio extraction, cutting, silence trim, concat |

## Known limits / tech debt
- Endpoints are synchronous and long-running (minutes); no job queue or progress endpoint.
- Sheet ↔ video link is the filename; deleting `Data/Original_videos/` breaks `create-*` for old sheets.
- No auth on the API; CORS is `*`. Default bind is localhost for that reason.
- `google-generativeai` is deprecated upstream in favour of `google-genai`; migration not done.
- `/process-video` does not check that Gemini returned one line per segment: if it drops or merges lines the Refined column is partly empty and those rows are skipped later.
- No automated tests yet.
