# Decision log

Model/agent identifiers are recorded only where known.

## 2026-11-06 — Folder cleanup and single entry point (done via Claude Code; exact model id for the edits not verified)
- **Problem:** 10+ near-identical backend copies (`Morning*.py`, `backend*.py`), unrelated projects, media and service-account keys in the repo; a 288 MB video in history made a GitHub push impossible.
- **Chosen:** `Morning_10.py` (latest by mtime; it has every endpoint of the earlier copies) became `main.py`; everything else moved to git-ignored `_archive/` (not deleted). Fresh git history (old one kept locally as branch `old-main`, never pushed).
- **Alternatives:** rewriting history with filter-repo (rejected: only one commit of value, a fresh start is simpler); deleting old files (rejected: a reversible move is cheaper than regret).
- **Trade-off:** old history is local-only; `_archive/` must be deleted manually when the owner is sure.

## 2026-11-06 — Behaviour fixes in `main.py`
- Clip start/end now come from the chosen segment itself. Previously `expand_segment_for_context` indexed the importance-sorted list with the avatar loop index, so clips could be cut from the wrong place.
- `avatar_id` / `voice_id` / `query` request fields are now actually used (were silently ignored).
- JSON sanitizer slices the outermost `[...]` (old code returned empty text for plain ``` fences, always triggering the fallback).
- Endpoints are sync `def` (thread pool) so long jobs don't freeze the server; Whisper model loaded once.
- Sheets shared as **reader** link instead of public **writer**; server binds to 127.0.0.1 by default.
- Gemini model made configurable (`GEMINI_MODEL`) because `gemini-1.5-flash` is retired.
- **Consequence:** public-link viewers can no longer edit the transcript; edit it as the owning service account, or change the role in `process_video`.
