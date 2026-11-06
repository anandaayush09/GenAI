"""Video-to-avatar-narration backend (FastAPI).

Pipeline: upload -> ffmpeg audio -> faster-whisper transcript -> Gemini cleanup ->
Google Sheet (human-editable) -> Gemini picks key segments -> HeyGen avatar intros ->
ffmpeg stitches each intro in front of its original clip.
Flow details: docs/agent/Architecture.md
"""
import functools
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid

import google.generativeai as genai
import requests
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure root logger once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
# Model ids get retired by Google; keep it overridable without a code change.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# HeyGen Configuration - ENHANCED
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")
HEYGEN_API_URL = "https://api.heygen.com/v2/video/generate"
HEYGEN_STATUS_URL = "https://api.heygen.com/v1/video_status.get"
DEFAULT_AVATAR_ID = os.getenv("HEYGEN_AVATAR_ID", "Abigail_expressive_2024112501")
DEFAULT_VOICE_ID = os.getenv("HEYGEN_VOICE_ID", "73c0b6a2e29d4d38aca41454bf58c955")

# Google Sheets setup with secure key path handling
key_path = os.getenv("GOOGLE_SA_KEY_PATH")
if not key_path or not os.path.isfile(key_path):
    raise FileNotFoundError("Google Service Account key file not found or path is not set correctly.")

with open(key_path) as f:
    sa_info = json.load(f)

logger.info(f"\U0001f510 Loaded Google credentials from: {key_path}")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(sa_info, scopes=SCOPES)
sheets_service = build('sheets', 'v4', credentials=credentials, cache_discovery=False)
drive_service = build('drive', 'v3', credentials=credentials)

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper functions ---
def call_gemini(prompt: str) -> str:
    model = genai.GenerativeModel(GEMINI_MODEL)
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise HTTPException(status_code=500, detail="AI model call failed.")

def extract_audio_with_ffmpeg(video_path, audio_path):
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1',
            audio_path
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed (code {e.returncode}): {e.stderr}")
        raise

def get_video_duration(path: str) -> float:
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ], capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

@functools.lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    # Loading the model takes seconds and ~hundreds of MB; do it once per process.
    return WhisperModel("base", device="cpu", compute_type="int8")

# --- AI-driven segment analysis ---

class SegmentAnalyzer:
    def __init__(self):
        self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)

    def _sanitize_gemini_json(self, text: str) -> str:
        """Keep only the outermost [...] of Gemini's reply (it may add prose or ``` fences)."""
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        start, end = text.find("["), text.rfind("]")
        return text[start:end + 1] if start != -1 and end > start else text.strip()

    def analyze_segments_for_avatar_commentary(self, segments: list, query: str = "Identify key topics") -> list:
        """Analyze segments and create avatar commentary scripts with AI-determined optimal length"""

        full_transcript = "\n".join([
            f"Segment {i}: {seg['text']} (Duration: {seg['end']-seg['start']:.1f}s)"
            for i, seg in enumerate(segments)
        ])

        prompt = f"""
        You are a friendly human presenter watching a video with the audience.
        Narrate each chosen segment in a natural, conversational way as if reacting to what you see on screen.

        Presenter brief from the user (use it to decide which segments matter):
        {query}

        Rules for narration:
        - Speak as if you are reacting to what the viewer is seeing in the footage.
        - Use conversational language and contractions (you're, it's, we'll, etc.).
        - Include extra background details, relevant facts, or quick examples to enrich the explanation.
        - If possible, add a short personal anecdote or analogy that makes the concept more relatable.
        - Keep the flow natural but aim for a fuller, richer narration — target 1.5–2× the length of a basic explanation.
        - Vary pacing and sentence length, include natural pauses (...), and occasional light commentary.
        - Reference the visuals directly ("Notice how...", "Right here you can see...").
        - Avoid repeating the exact words from the original segment.

        IMPORTANT: You must return ONLY a valid JSON array.
        Each object in the array must strictly have these keys:
        - segment_index (integer)
        - importance_score (1–10 integer)
        - avatar_script (string)
        - estimated_duration (integer seconds)
        - script_complexity ("simple"|"moderate"|"complex"|"comprehensive")
        - content_depth ("surface"|"moderate"|"deep"|"comprehensive")
        - topic_summary (string)
        - why_important (string)
        - educational_value (string)

        All string values must be JSON-safe: escape internal quotes (\") and remove unescaped newlines.

        Transcript:
        {full_transcript}
        """

        try:
            response = self.gemini_model.generate_content(prompt)
            analysis_text = response.text.strip()

            # Sanitize before parsing
            clean_text = self._sanitize_gemini_json(analysis_text)

            try:
                # Use the relaxed parser to handle newlines within string values
                analysis = json.loads(clean_text, strict=False) 
            except json.JSONDecodeError as e:
                # This block will now only catch more severe, unrecoverable JSON errors
                logger.error(f"[SANITIZER] JSON parsing failed even after cleaning: {e}")
                # For debugging, it's helpful to see what caused the failure
                logger.debug(f"--- FAILED JSON TEXT ---\n{clean_text}\n-------------------------")
                return self._create_enhanced_fallback(segments)

            enhanced_analysis = []
            for item in analysis:
                if 0 <= item['segment_index'] < len(segments):
                    segment = segments[item['segment_index']]
                    enhanced_item = {
                        **item,
                        'avatar_description': item['avatar_script'],
                        'start_time': segment['start'],
                        'end_time': segment['end'],
                        'original_text': segment['text'],
                        'dynamic_length': True,
                        'ai_optimized': True
                    }
                    enhanced_analysis.append(enhanced_item)

            logger.info(f"[ANALYZER] Selected {len(enhanced_analysis)} segments with AI-determined lengths")
            for item in enhanced_analysis:
                logger.info(f"  - Segment {item['segment_index']}: {item['estimated_duration']}s ({item['script_complexity']} complexity)")

            return sorted(enhanced_analysis, key=lambda x: -x['importance_score'])[:6]

        except Exception as e:
            logger.error(f"Enhanced segment analysis failed: {e}")
            return self._create_enhanced_fallback(segments)

    
    def _create_enhanced_fallback(self, segments: list) -> list:
        """Create enhanced fallback with varied script lengths based on content analysis"""
        fallback = []
        for i in range(0, len(segments), max(1, len(segments)//4)):
            if i < len(segments):
                # Analyze content length to determine script complexity
                content_length = len(segments[i]['text'])
                duration = segments[i]['end'] - segments[i]['start']
                
                if content_length > 200 or duration > 30:
                    script_length = "comprehensive"
                    estimated_duration = 60
                    script = f"In this segment, we explore a crucial topic that builds on earlier insights. Here's what's coming up: {segments[i]['text'][:60]}..."
                elif content_length > 100 or duration > 15:
                    script_length = "moderate"
                    estimated_duration = 30
                    script = f"This part introduces key ideas that will shape your understanding. Here's what to expect: {segments[i]['text'][:60]}..."
                else:
                    script_length = "simple"
                    estimated_duration = 15
                    script = f"Here's a short but important insight to pay attention to: {segments[i]['text'][:50]}..."

                                
                fallback.append({
                    'segment_index': i,
                    'importance_score': 6,
                    'avatar_description': script,
                    'estimated_duration': estimated_duration,
                    'script_complexity': script_length,
                    'content_depth': script_length,
                    'topic_summary': segments[i]['text'][:50],
                    'why_important': f'Fallback selection with {script_length} complexity based on content analysis',
                    'educational_value': f'Provides {script_length} introduction to enhance comprehension',
                    'start_time': segments[i]['start'],
                    'end_time': segments[i]['end'],
                    'original_text': segments[i]['text'],
                    'dynamic_length': True,
                    'ai_optimized': False
                })
        return fallback[:5]  # Max 5 segments for fallback

# --- ENHANCED: HeyGen Integration with Dynamic Script Handling ---
class HeyGenAvatarGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Api-Key": api_key
        }
    
    def generate_avatar_video(self, script: str, output_path: str, avatar_id: str = None, 
                            voice_id: str = None, estimated_duration: int = None) -> str:
        """Generate avatar video with optimized settings for dynamic script lengths"""
        avatar_id = avatar_id or DEFAULT_AVATAR_ID
        voice_id = voice_id or DEFAULT_VOICE_ID
        
        # Analyze script for optimal generation settings
        script_analysis = self._analyze_script_requirements(script, estimated_duration)
        
        logger.info(f"[HEYGEN] Generating {script_analysis['complexity']} avatar video...")
        logger.info(f"[HEYGEN] Estimated duration: {script_analysis['estimated_duration']}s")
        logger.info(f"[HEYGEN] Script preview: {script[:150]}...")
        
        # Try different configurations based on script complexity
        generation_configs = self._get_generation_configs(script_analysis)
        
        for config in generation_configs:
            try:
                payload = {
                    "video_inputs": [{
                        "character": {
                            "type": "avatar",
                            "avatar_id": avatar_id,
                            "avatar_style": config["avatar_style"]
                        },
                        "voice": {
                            "type": "text",
                            "input_text": script,
                            "voice_id": voice_id,
                            "voice_settings": config.get("voice_settings", {})
                        }
                    }],
                    "dimension": config["dimension"],
                    "quality": config.get("quality", "medium")
                }
                
                response = requests.post(HEYGEN_API_URL, headers=self.headers, json=payload)
                response.raise_for_status()
                
                data = response.json().get("data", {})
                if "video_id" not in data:
                    logger.warning(f"Config failed: {config['name']}")
                    continue
                
                video_id = data["video_id"]
                logger.info(f"[HEYGEN] Video generation started (ID: {video_id}) with {config['name']}")
                
                # Poll for completion with extended timeout for longer scripts
                max_wait_time = 600  # Force wait up to 10 minutes  # Dynamic timeout
                video_url = self._poll_for_completion(video_id, max_wait_time)
                
                if video_url:
                    return self._download_video(video_url, output_path)
                    
            except Exception as e:
                logger.error(f"Failed with config {config['name']}: {e}")
                continue
        
        raise Exception("All HeyGen generation configurations failed")
    
    def _analyze_script_requirements(self, script: str, estimated_duration: int = None) -> dict:
        """Analyze script to determine optimal generation requirements"""
        word_count = len(script.split())
        char_count = len(script)
        
        # Estimate duration if not provided (average speaking rate: 150-160 words per minute)
        if not estimated_duration:
            estimated_duration = max(10, (word_count / 150) * 60)
        
        # Determine complexity based on multiple factors
        if word_count > 200 or estimated_duration > 90:
            complexity = "comprehensive"
        elif word_count > 100 or estimated_duration > 45:
            complexity = "complex"
        elif word_count > 50 or estimated_duration > 20:
            complexity = "moderate"
        else:
            complexity = "simple"
        
        return {
            "word_count": word_count,
            "char_count": char_count,
            "estimated_duration": estimated_duration,
            "complexity": complexity,
            "requires_high_quality": complexity in ["complex", "comprehensive"],
            "requires_extended_timeout": estimated_duration > 60
        }
    
    def _get_generation_configs(self, script_analysis: dict) -> list:
        """Force only 360p generation"""
        return [
            {
                "name": "Forced-360p-Only",
                "dimension": {"width": 640, "height": 360},
                "avatar_style": "normal",
                "quality": "medium"
            }
        ]

    def _poll_for_completion(self, video_id: str, max_wait_seconds: int = 600):
        """Poll HeyGen API for video completion with dynamic timeout"""
        status_url = f"{HEYGEN_STATUS_URL}?video_id={video_id}"
        max_attempts = max_wait_seconds // 5  # Poll every 5 seconds
        
        logger.info(f"[HEYGEN] Polling for up to {max_wait_seconds}s ({max_attempts} attempts)")
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(status_url, headers=self.headers)
                response.raise_for_status()
                
                data = response.json().get("data", {})
                status = data.get("status")
                
                if attempt % 6 == 0:  # Log every 30 seconds
                    logger.info(f"[HEYGEN] Poll #{attempt+1}/{max_attempts}: Status = {status}")
                
                if status == "completed":
                    logger.info(f"[HEYGEN] Generation completed after {(attempt+1)*5}s")
                    return data.get("video_url")
                elif status == "failed":
                    logger.error(f"Generation failed: {data}")
                    return None
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Polling error: {e}")
                time.sleep(5)
        
        logger.error(f"Video generation timed out after {max_wait_seconds}s")
        return None
    
    def _download_video(self, video_url: str, output_path: str) -> str:
        """Download generated video with progress tracking"""
        try:
            logger.info(f"[HEYGEN] Downloading video from {video_url}")
            
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress for large files
                        if total_size > 0 and downloaded % (max(1, total_size // 10)) == 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"[HEYGEN] Download progress: {progress:.1f}%")
            
            logger.info(f"[HEYGEN] Video saved to {output_path} ({downloaded/1024/1024:.1f}MB)")
            return output_path
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

# --- ENHANCED: Advanced Video Compositor ---
class AvatarVideoComposer:
    def __init__(self):
        self.temp_dir = "./Data/temp_compositions"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def create_matthew_berman_style_video(self, original_video: str, analyzed_segments: list, 
                                        avatar_generator: HeyGenAvatarGenerator, output_path: str,
                                        avatar_id: str = None, voice_id: str = None) -> str:
        """Create video in Matthew Berman style with AI-optimized dynamic-length avatar introductions"""
        
        logger.info(f"[COMPOSER] Creating Matthew Berman style video with {len(analyzed_segments)} AI-optimized segments")
        
        # Step 1: Generate all avatar introduction videos with dynamic lengths
        avatar_videos = []
        total_avatar_time = 0
        
        for i, segment_data in enumerate(analyzed_segments):
            avatar_filename = f"avatar_intro_{i}_{uuid.uuid4().hex[:8]}.mp4"
            avatar_path = os.path.join(self.temp_dir, avatar_filename)
            
            try:
                logger.info(f"[COMPOSER] Generating avatar {i+1}/{len(analyzed_segments)}: "
                          f"{segment_data['script_complexity']} complexity, "
                          f"~{segment_data['estimated_duration']}s duration")
                
                avatar_generator.generate_avatar_video(
                    script=segment_data['avatar_description'],
                    output_path=avatar_path,
                    avatar_id=avatar_id,
                    voice_id=voice_id,
                    estimated_duration=segment_data.get('estimated_duration')
                )
                
                actual_duration = get_video_duration(avatar_path)
                total_avatar_time += actual_duration
                
                avatar_videos.append({
                    'path': avatar_path,
                    'duration': actual_duration,
                    'estimated_duration': segment_data.get('estimated_duration', 0),
                    'complexity': segment_data['script_complexity'],
                    'segment_data': segment_data
                })
                
                logger.info(f"[COMPOSER] Avatar {i+1} completed: {actual_duration:.1f}s actual vs "
                          f"{segment_data.get('estimated_duration', 0)}s estimated")
                
            except Exception as e:
                logger.error(f"Failed to generate avatar video {i}: {e}")
                continue
        
        if not avatar_videos:
            raise Exception("No avatar videos were generated successfully")
        
        logger.info(f"[COMPOSER] Generated {len(avatar_videos)} avatar videos, "
                  f"total introduction time: {total_avatar_time:.1f}s")
        
        # Step 2: Create video segments from original video with context buffer
        video_segments = []
        original_duration = get_video_duration(original_video)

        for i, avatar_data in enumerate(avatar_videos):
            segment_info = avatar_data['segment_data']
            segment_filename = f"segment_{i}_{uuid.uuid4().hex[:8]}.mp4"
            segment_path = os.path.join(self.temp_dir, segment_filename)
            
            # Clip bounds come from the segment itself. (Never index analyzed_segments by i:
            # it is sorted by importance and avatar_videos can be shorter after HeyGen failures.)
            buffer_start, buffer_end = segment_info['start_time'], segment_info['end_time']

            # Ensure minimum segment duration for visual flow
            duration = buffer_end - buffer_start
            if duration < 10.0:  # Now minimum 10 seconds
                needed_extension = 10.0 - duration
                buffer_start = max(0, buffer_start - needed_extension / 2)
                buffer_end = min(original_duration, buffer_end + needed_extension / 2)
                duration = buffer_end - buffer_start

            
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(buffer_start),
                "-i", original_video,
                "-t", str(duration),
                "-avoid_negative_ts", "make_zero",
                "-fflags", "+genpts",
                "-c:v", "libx264", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k", "-ar", "44100",  # Ensure consistent audio
                "-af", "afade=in:st=0:d=0.3,afade=out:st={}:d=0.3".format(duration-0.3),  # Audio fades
                segment_path
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Verify segment has audio
                audio_check_cmd = [
                    "ffprobe", "-v", "error", "-select_streams", "a:0", 
                    "-show_entries", "stream=duration", "-of", "csv=p=0", segment_path
                ]
                audio_result = subprocess.run(audio_check_cmd, capture_output=True, text=True)
                
                if audio_result.returncode == 0 and audio_result.stdout.strip():
                    video_segments.append({
                        'path': segment_path,
                        'duration': duration,
                        'avatar_video': avatar_data,
                        'has_audio': True
                    })
                    logger.info(f"[COMPOSER] Extracted segment {i+1}: {duration:.1f}s with audio")
                else:
                    logger.warning(f"[COMPOSER] Segment {i+1} missing audio, skipping")
                    
            except Exception as e:
                logger.error(f"Failed to extract video segment {i}: {e}")
                continue

        # Step 3: Create final composition using FFmpeg
        return self._compose_final_video(video_segments, output_path)
    
    def _compose_final_video(self, video_segments: list, output_path: str) -> str:
        """Compose final video with proper audio handling and crossfade transitions"""
    
        # Create individual video files with crossfade for smooth transitions
        processed_segments = []
        
        for i, segment in enumerate(video_segments):
            # First trim avatar video audio
            avatar_path = self._trim_avatar_audio_silence(segment['avatar_video']['path'])
            
            # Create smooth transition between avatar and original video
            transition_filename = f"transition_{i}_{uuid.uuid4().hex[:8]}.mp4"
            transition_path = os.path.join(self.temp_dir, transition_filename)
            
            # Create filter complex for smooth audio/video transition
            filter_complex = [
                f"[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[outv][outa]",
                f"[outa]afade=in:st=0:d=0.2[audio_out]"
            ]
            
            cmd = [
                "ffmpeg", "-y",
                "-i", avatar_path,
                "-i", segment['path'],
                "-filter_complex", ";".join(filter_complex),
                "-map", "[outv]", "-map", "[audio_out]",
                "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
                "-movflags", "+faststart",
                transition_path
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                processed_segments.append(transition_path)
                logger.info(f"[COMPOSER] Created transition segment {i+1}")
            except Exception as e:
                logger.error(f"Failed to create transition segment {i}: {e}")
                continue
        
        if not processed_segments:
            raise Exception("No segments processed successfully")
        
        # Final concatenation of all transition segments
        concat_file = os.path.join(self.temp_dir, f"concat_{uuid.uuid4().hex[:8]}.txt")
        
        with open(concat_file, 'w') as f:
            for segment_path in processed_segments:
                f.write(f"file '{os.path.abspath(segment_path)}'\n")
        
        # Final composition with audio stream mapping
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            "-map", "0:v:0", "-map", "0:a:0",  # Explicit stream mapping
            "-movflags", "+faststart",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"[COMPOSER] Final video created: {output_path}")
            
            # Cleanup temporary files
            self._cleanup_temp_files([concat_file] + processed_segments +
                                   [s['path'] for s in video_segments] + 
                                   [s['avatar_video']['path'] for s in video_segments])
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg composition failed: {e.stderr}")
            raise Exception(f"Video composition failed: {e.stderr}")

    def _trim_avatar_audio_silence(self, video_path: str) -> str:
        """Trim trailing silence from avatar video"""
        try:
            # Detect silence at the end
            detect_cmd = [
                "ffmpeg", "-i", video_path,
                "-af", "silencedetect=noise=-30dB:duration=0.5",
                "-f", "null", "-", "-hide_banner"
            ]
            result = subprocess.run(detect_cmd, capture_output=True, text=True)
            
            # Find last silence start
            lines = result.stderr.split('\n')
            last_silence_start = None
            
            for line in lines:
                if "silence_start" in line:
                    try:
                        last_silence_start = float(line.split("silence_start: ")[1].split()[0])
                    except (IndexError, ValueError):
                        continue
            
            if last_silence_start and last_silence_start > 2.0:
                trimmed_path = video_path.replace('.mp4', '_trimmed.mp4')
                
                trim_cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-t", str(last_silence_start + 0.5),
                    "-af", "afade=out:st={}:d=0.3".format(last_silence_start + 0.2),
                    "-c:v", "copy", "-c:a", "aac",
                    trimmed_path
                ]
                
                subprocess.run(trim_cmd, check=True, capture_output=True)
                # Replace the original with the trimmed version for subsequent steps
                os.remove(video_path)
                os.rename(trimmed_path, video_path)
                logger.info(f"[COMPOSER] Trimmed avatar video to {last_silence_start + 0.5:.1f}s")

        except Exception as e:
            logger.warning(f"Avatar trimming failed: {e}")
        
        return video_path
    
    def _cleanup_temp_files(self, file_paths: list):
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")

# --- ENHANCED API ENDPOINTS ---

@app.post("/process-video")
def process_video(file: UploadFile = File(...)):
    try:
        t0 = time.time()
        logger.info(f" Start upload @ {t0:.3f}")
        
        uid = str(uuid.uuid4())
        # Ensure filename is filesystem-safe
        safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', file.filename)
        filename = f"{os.path.splitext(safe_filename)[0]}_{uid}.mp4"

        uploads_dir = "./Data/Original_videos"
        audio_dir = "./Data/Extracted_Audio"
        
        os.makedirs(uploads_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        
        video_path = os.path.join(uploads_dir, filename).replace("\\", "/")
        audio_path = os.path.join(audio_dir, filename.replace('.mp4', '.wav')).replace("\\", "/")
        
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)  # stream; don't hold the video in RAM
        
        logger.info(f" Saved video Δ {time.time() - t0:.3f}s")
        
        if os.path.getsize(video_path) < 100000:
            raise HTTPException(status_code=400, detail="Uploaded file too small or corrupted.")
        
        extract_audio_with_ffmpeg(video_path, audio_path)
        logger.info(f" Audio extracted")
        
        # --- Transcribe with Whisper ---
        segments_generator, _ = get_whisper_model().transcribe(audio_path, no_speech_threshold=2.0, language="en")
        segments = [{"start": seg.start, "end": seg.end, "text": seg.text} for seg in segments_generator]
        
        logger.info(f" Transcription complete with {len(segments)} segments.")
        
        # --- Prepare prompt for Gemini refinement ---
        prompt = """
        You are given a list of transcript segments transcribed by a whisper model, each prefixed by its timestamp.
        Your task is to refine each segment individually with these strict rules:
        
        1. Preserve the exact number of segments. Do not delete, merge, or split any segments.
        2. Minimal changes only: Remove only filler words ("So", "okay", "ok", "um," "uh," "like," "you know"). Correct obvious spelling mistakes or minor grammar errors.
        3. Maintain timestamps: Output exactly one line per input segment. Each line must begin with the original timestamp, a space, then the refined text.
        4. If a segment is empty or contains only a filler word, return it as is but without the filler word if possible.
        
        Now refine the following segments:\n
        """
        
        rows = [["SegmentID", "Start", "End", "New Start", "New End", "Original", "Refined",
                "Pause at end(sec)", "Audio Length", "Video Length", "Flag", "Clone Voice",
                "Video Speed flag", "Out of sync", "Additional comments"]]
        
        current_t = 0
        segment_id_counter = 0
        
        for seg in segments:
            if seg["start"] - current_t > 0.2:
                rows.append([
                    segment_id_counter, current_t, seg["start"], current_t, seg["start"],
                    "", "", 0, seg["start"] - current_t, seg["start"] - current_t,
                    "SILENCE", "yes", "", "", ""
                ])
                prompt += f"{current_t}\n"
                segment_id_counter += 1
            
            prompt += f"{seg['start']} {seg['text'].strip()}\n"
            rows.append([
                segment_id_counter, seg["start"], seg["end"], "", "", seg["text"].strip(),
                "", 0, "", seg["end"] - seg["start"], "", "yes", "", "", ""
            ])
            current_t = seg["end"]
            segment_id_counter += 1
        
        duration = get_video_duration(video_path)
        if duration - current_t > 1:
            rows.append([
                segment_id_counter, current_t, duration, current_t, duration,
                "", "", 0, duration - current_t, duration - current_t,
                "SILENCE", "yes", "", "", ""
            ])
        
        logger.info(f" Prompt & rows constructed")
        
        # --- Call Gemini ---
        refined_lines = call_gemini(prompt).splitlines()
        logger.info(f" Gemini responded with {len(refined_lines)} lines")
        
        refined_map = {}
        for line in refined_lines:
            try:
                timestamp_str, text = line.split(' ', 1)
                refined_map[float(timestamp_str)] = text
            except ValueError:
                continue
        
        for i in range(1, len(rows)):
            start_time = rows[i][1]
            if isinstance(start_time, (int, float)) and start_time in refined_map:
                # Only update the 'Refined' column (index 6)
                rows[i][6] = refined_map[start_time]

        
        # --- Upload to Google Sheet ---
        sheet_title = os.path.basename(video_path)

        sheet = sheets_service.spreadsheets().create(
            body={
                'properties': {'title': sheet_title},
                'sheets': [],
            },
            fields='spreadsheetId'
        ).execute()
        

        sheet_id = sheet['spreadsheetId']
        
        drive_service.permissions().create(
            fileId=sheet_id, body={'type': 'anyone', 'role': 'reader'}
        ).execute()
        
        sheets_service.spreadsheets().values().update(
            spreadsheetId=sheet_id, range='A1',
            valueInputOption='RAW', body={'values': rows}
        ).execute()
        
        logger.info(f" Sheet created and populated")
        os.remove(audio_path)
        
        return JSONResponse({
            "message": "Video processed successfully",
            "spreadsheetId": sheet_id,
            "spreadsheet_url": f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit",
            "segments_count": len(segments)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in process_video endpoint")
        raise HTTPException(status_code=500, detail="Internal server error during video processing.")

@app.get("/fetch-segments")
def fetch_segments(sheetId: str):
    try:
        resp = sheets_service.spreadsheets().values().get(
            spreadsheetId=sheetId, range='A2:G'
        ).execute()
        values = resp.get('values', [])
        segments = [
            {'id': row[0], 'start': row[1], 'end': row[2], 'original': row[5], 'refined': row[6]}
            for row in values if len(row) > 6
        ]
        return JSONResponse({"segments": segments})
    except HttpError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- ENHANCED: Matthew Berman Style Video Creation with AI-Driven Dynamic Length ---
@app.post("/create-matthew-berman-style-video")
def create_matthew_berman_style_video(
    sheetId: str = Body(..., embed=True),
    query: str = Body(
        "Analyze and create comprehensive introductions for key segments",
        embed=True
    ),
    avatar_id: str = Body(DEFAULT_AVATAR_ID, embed=True),
    voice_id: str = Body(DEFAULT_VOICE_ID, embed=True)
):
    """
    Creates a Matthew Berman style video with AI-driven dynamic-length avatar introductions.
    The AI determines optimal introduction length based on content complexity and educational value.
    """
    
    if not HEYGEN_API_KEY:
        raise HTTPException(status_code=500, detail="HeyGen API key not configured")
    
    try:
        t0 = time.time()
        logger.info(f"[DYNAMIC-STYLE] Starting AI-driven dynamic length video creation for sheet: {sheetId}")
        
        # 1. Fetch transcript data from Google Sheets
        try:
            spreadsheet = sheets_service.spreadsheets().get(spreadsheetId=sheetId).execute()
            filename = spreadsheet['properties']['title']
            
            resp = sheets_service.spreadsheets().values().get(spreadsheetId=sheetId, range='A2:G').execute()
            values = resp.get('values', [])
            
            if not values:
                raise HTTPException(status_code=404, detail="No transcript data found")
            
            # Extract segments with refined text
            segments = []
            for row in values:
                if len(row) > 6 and row[6].strip():
                    try:
                        segments.append({
                            'start': float(row[1]),
                            'end': float(row[2]),
                            'text': row[6].strip()
                        })
                    except (ValueError, IndexError):
                        continue
            
            if not segments:
                raise HTTPException(status_code=400, detail="No valid segments found")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch transcript data: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch transcript data")
        
        # 2. Analyze segments for AI-driven dynamic avatar commentary
        analyzer = SegmentAnalyzer()
        analyzed_segments = analyzer.analyze_segments_for_avatar_commentary(segments, query)
        
        if not analyzed_segments:
            raise HTTPException(status_code=400, detail="No segments selected for commentary")
        
        # Log AI analysis results
        logger.info(f"[DYNAMIC-STYLE] AI selected {len(analyzed_segments)} segments:")
        total_intro_time = 0
        for i, seg in enumerate(analyzed_segments):
            duration = seg.get('estimated_duration', 0)
            total_intro_time += duration
            logger.info(f"  {i+1}. {seg['script_complexity']} ({duration}s): {seg['topic_summary']}")
        
        logger.info(f"[DYNAMIC-STYLE] Total estimated introduction time: {total_intro_time}s")
        
        # 3. Initialize components
        avatar_generator = HeyGenAvatarGenerator(HEYGEN_API_KEY)
        composer = AvatarVideoComposer()
        
        # 4. Find original video file
        original_video_path = os.path.join("./Data/Original_videos", filename).replace("\\", "/")
        if not os.path.exists(original_video_path):
            raise HTTPException(status_code=404, detail=f"Original video file not found: {original_video_path}")
        
        # 5. Create output path
        output_dir = "./Data/Final_videos"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"ai_dynamic_{os.path.basename(filename)}"
        output_path = os.path.join(output_dir, output_filename).replace("\\", "/")
        
        # 6. Generate the AI-optimized Matthew Berman style video
        final_video_path = composer.create_matthew_berman_style_video(
            original_video=original_video_path,
            analyzed_segments=analyzed_segments,
            avatar_generator=avatar_generator,
            output_path=output_path,
            avatar_id=avatar_id,
            voice_id=voice_id
        )
        
        t1 = time.time()
        logger.info(f"[DYNAMIC-STYLE] Complete! Total time: {t1-t0:.2f}s")
        
        return JSONResponse({
            "message": "AI-driven dynamic length Matthew Berman style video created successfully!",
            "final_video_path": final_video_path,
            "segments_used": len(analyzed_segments),
            "total_intro_time_estimated": total_intro_time,
            "processing_time_seconds": round(t1-t0, 2),
            "ai_analysis": [
                {
                    "segment": i + 1,
                    "topic": seg['topic_summary'],
                    "complexity": seg['script_complexity'],
                    "estimated_duration": seg['estimated_duration'],
                    "importance_score": seg['importance_score'],
                    "educational_value": seg['educational_value'],
                    "script_preview": seg['avatar_description'][:150] + "..." if len(seg['avatar_description']) > 150 else seg['avatar_description'],
                    "start_time": seg['start_time'],
                    "end_time": seg['end_time'],
                    "segment_duration": round(seg['end_time'] - seg['start_time'], 2)
                }
                for i, seg in enumerate(analyzed_segments)
            ]
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in create_matthew_berman_style_video endpoint")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# --- Legacy compatibility endpoint ---
@app.post("/create-highlight-video") 
def create_highlight_video(sheetId: str, query: str = Body(..., embed=True)):
    """Legacy endpoint - now creates AI-driven dynamic length videos by default"""
    return create_matthew_berman_style_video(sheetId, query, DEFAULT_AVATAR_ID, DEFAULT_VOICE_ID)

if not HEYGEN_API_KEY:
    logger.warning("HEYGEN_API_KEY not set: video creation endpoints will return 500")

if __name__ == "__main__":
    import uvicorn
    # Localhost by default: the API has no auth and spends HeyGen credits. Set HOST=0.0.0.0 to expose it.
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
