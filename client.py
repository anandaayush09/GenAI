#!/usr/bin/env python3
"""Upload a video to the local backend, then ask it to build the avatar-narrated cut.

Usage: python client.py path/to/video.mp4   (start the server first: python main.py)
"""
import os
import sys

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

# Instruction passed to Gemini when it picks segments and writes the narration.
NARRATION_QUERY = (
    "Identify the most important segments in the transcript and write narration that sounds like "
    "a friendly human presenter is guiding the viewer while watching the footage together.\n\n"
    "- Use casual, conversational language.\n"
    "- Refer directly to what's happening in the video at that moment.\n"
    "- Flow naturally into explanations without rigid structure.\n"
    "- Include small pauses, varied pacing, and occasional light commentary.\n"
    "- Make it feel spontaneous and reactive, not scripted."
)


def fail(msg: str, resp: requests.Response) -> None:
    print(f"{msg}: {resp.status_code}\n{resp.text}")
    sys.exit(1)


def upload_video(path: str) -> str:
    """POST the video to /process-video; returns the transcript spreadsheetId."""
    print(f"Uploading {path}")
    with open(path, "rb") as f:
        encoder = MultipartEncoder(fields={"file": (os.path.basename(path), f, "video/mp4")})
        resp = requests.post(f"{API_BASE}/process-video", data=encoder,
                             headers={"Content-Type": encoder.content_type})
    if resp.status_code not in (200, 202):
        fail("Upload failed", resp)
    data = resp.json()
    print("Upload OK:", data)
    if not data.get("spreadsheetId"):
        fail("No spreadsheetId returned", resp)
    return data["spreadsheetId"]


def create_avatar_narration(sheet_id: str) -> str:
    """POST to /create-matthew-berman-style-video; returns the final video path."""
    print(f"Creating narrated video for sheet {sheet_id} (can take several minutes)")
    resp = requests.post(f"{API_BASE}/create-matthew-berman-style-video",
                         json={"sheetId": sheet_id, "query": NARRATION_QUERY})
    if resp.status_code != 200:
        fail("Video creation failed", resp)
    result = resp.json()
    print("Result:", result)
    return result.get("final_video_path", "")


def main() -> None:
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        sys.exit("Usage: python client.py path/to/video.mp4")
    final_path = create_avatar_narration(upload_video(sys.argv[1]))
    print(f"Done. Final video: {final_path}")


if __name__ == "__main__":
    main()
