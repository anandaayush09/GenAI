# Handover

## State (2026-11-06)
Repo cleaned for GitHub: `main.py` (server), `client.py`, `requirements.txt`, `.env.example`, `README.md`, `docs/agent/`. Old copies live in the local, git-ignored `_archive/`.

## Done this session
- Archived clutter; renamed `Morning_10.py` → `main.py`, `video_upload_client.py` → `client.py`.
- Fixed the bugs listed in `Decisions.md`; trimmed `requirements.txt` to imported packages (added `python-multipart`, which uploads need).
- Prepared a fresh orphan git history (nothing pushed).

## Verification
A static check, a compile, a server start and a sanitizer check were run; a full end-to-end run (uses HeyGen credits and creates a real Google Sheet) has NOT been done after the edits. Anything not stated as run is untested.

## Next steps
1. Run `python main.py` + `python client.py <short video>` end to end and confirm clips line up with the narration.
2. Rotate the HeyGen key (it sat in plaintext in old scripts, now in `_archive/avatars.py`) and the service-account key if either was ever shared.
3. Review the staged files, commit, then `git remote add origin <url>` and `git push -u origin main`.
4. Optional: migrate to `google-genai`; add a job queue; add tests for the JSON sanitizer and clip bounds.

## Warnings
- `.env` and `google_sa_key.json` must stay untracked.
- Branch `old-main` contains the 288 MB video and two private keys: never push it.
