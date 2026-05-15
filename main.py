"""
Clinical AI Assistant — Tool Call Tester
-----------------------------------------
Tests tool calling with a local Ollama model via text prompts or voice input.
Results are saved to session logs (JSON + plain text summary).
Tool output (patient logs, reminders) is persisted to SQLite under data/.

Usage:
    python main.py                              # Text mode, default model
    python main.py --voice                      # Voice input (Faster-Whisper STT)
    python main.py --voice --stt-model tiny     # Faster STT (less accurate)
    python main.py --model qwen3.5:0.8b || granite4:350m      # Different LLM
    python main.py --list-tools                 # Show available tools
    python main.py --no-tts                     # Disable TTS output
"""

import argparse
import io
import json
import os
import tempfile
import re
import sqlite3
import subprocess
import shutil
import sys
import threading
import queue
import uuid
import wave
from datetime import datetime, timedelta
from pathlib import Path

import ollama

# ── Optional: Piper TTS ───────────────────────────────────────────────────────
try:
    from piper.voice import PiperVoice
    _PIPER_AVAILABLE = True
except ImportError:
    _PIPER_AVAILABLE = False

# ── Optional: Faster-Whisper STT ─────────────────────────────────────────────
try:
    from faster_whisper import WhisperModel
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False

# ── Optional: sounddevice (microphone capture) ────────────────────────────────
try:
    import sounddevice as sd
    import numpy as np
    _SOUNDDEVICE_AVAILABLE = True
except ImportError:
    _SOUNDDEVICE_AVAILABLE = False

# ── Optional: display.py OLED integration ────────────────────────────────────
try:
    from display import DisplayController
    _OLED_AVAILABLE = True
except Exception as e:
    print(f"[OLED] Display unavailable: {e}")
    _OLED_AVAILABLE = False


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

DEFAULT_MODEL   = "granite4:350m"
DEFAULT_STT     = "base"          # Whisper model size: tiny | base | small
LOGS_DIR        = Path("logs")
DATA_DIR        = Path("data")
TOOLS_FILE      = Path("tools.json")
VOICES_DIR      = Path("voices")
DEFAULT_VOICE   = VOICES_DIR / "en_US-lessac-medium.onnx"

# Microphone recording parameters
_STT_SAMPLE_RATE      = 16_000   # Hz  — Whisper expects 16 kHz mono
_STT_CHUNK_FRAMES     = 512      # ~32 ms per chunk at 16 kHz
_STT_SILENCE_SECONDS  = 1.5      # stop recording after this much silence
_STT_ENERGY_THRESHOLD = 0.01     # RMS threshold to distinguish speech from noise
_STT_MAX_SECONDS      = 30       # hard cap per utterance

SYSTEM_PROMPT = """You are a clinical AI assistant supporting nurses and doctors in an active ward environment.

You have access to tools for:
  • Logging patient visits and clinical notes (log_patient)
  • Retrieving today's or a shift's patient list (get_patients_today)
  • Looking up all notes recorded for a specific patient (get_patient_notes)
  • Updating or correcting an existing patient note (update_patient_note)
  • Deleting an incorrect log entry by ID (delete_patient_log)
  • Setting timed reminders and alerts (set_reminder)
  • Generating shift handover summaries (summarise_shift)

Behavioural rules:
  1. ALWAYS call the appropriate tool when the request matches a tool's purpose — never answer from memory alone.
  2. After receiving a tool result, give a DETAILED, clinical-quality response:
       - For patient notes: repeat the note verbatim and add context (time logged, date, entry ID).
       - For patient lists: name every patient, their time, and their note.
       - For reminders: confirm exact trigger time and what will be reminded.
       - For summaries: present a structured handover-style report.
  3. If a log entry was just created, always confirm the patient name, note, time, date, and entry ID back to the user.
  4. If the user wants to retrieve notes for a specific patient, use get_patient_notes — NOT get_patients_today.
  5. Be professional, clear, and thorough. Clinical accuracy is critical."""


# ─────────────────────────────────────────────
# Tool Definitions  (loaded from tools.json)
# ─────────────────────────────────────────────

def load_tools() -> list:
    """Load tool definitions from tools.json."""
    if not TOOLS_FILE.exists():
        print(f"[ERROR] {TOOLS_FILE} not found. Cannot load tool definitions.")
        sys.exit(1)
    with open(TOOLS_FILE, "r") as f:
        return json.load(f)


TOOLS = load_tools()


# ─────────────────────────────────────────────
# Speech-to-Text  (Faster-Whisper + sounddevice)
# ─────────────────────────────────────────────

class SpeechListener:
    """
    Records microphone audio with voice-activity detection (VAD) and
    transcribes it with Faster-Whisper.

    The recorder captures audio in small chunks and tracks an RMS energy
    level.  Recording starts immediately; it stops automatically once a
    period of silence longer than _STT_SILENCE_SECONDS follows at least
    one chunk of detected speech.
    """

    def __init__(self, model_size: str = DEFAULT_STT, device: int | str | None = None, display=None):
        self.display = display
        if not _WHISPER_AVAILABLE:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Run:  pip install faster-whisper --break-system-packages"
            )
        if not _SOUNDDEVICE_AVAILABLE:
            raise RuntimeError(
                "sounddevice / numpy are not installed. "
                "Run:  pip install sounddevice numpy --break-system-packages"
            )

        print(f"[STT] Loading Whisper '{model_size}' model … ", end="", flush=True)
        # int8 quantisation keeps memory and CPU use low on the Pi
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("ready.")

        self._device      = device
        self._sample_rate = _STT_SAMPLE_RATE

        # Verify sample rate; fallback to device default if 16kHz fails (e.g. on some RPi mics)
        try:
            sd.check_input_settings(device=self._device, samplerate=self._sample_rate, channels=1)
        except Exception:
            try:
                info = sd.query_devices(self._device, 'input')
                self._sample_rate = int(info['default_samplerate'])
                print(f"\n[STT] Note: 16kHz not supported natively. Using {self._sample_rate}Hz (will resample).")
            except Exception as e:
                print(f"\n[STT] Warning: Could not query device settings: {e}")

        self._chunk_frames = _STT_CHUNK_FRAMES

    # ── Public API ────────────────────────────────────────────────────

    def listen(self) -> str | None:
        """
        Block until the user speaks and falls silent, then return the
        transcribed text.  Returns None if nothing intelligible was heard.
        """
        audio_chunks = self._record_utterance()
        if not audio_chunks:
            return None

        audio_np = np.concatenate(audio_chunks, axis=0).flatten().astype(np.float32)

        # Resample to 16kHz if the hardware used a different rate
        if self._sample_rate != 16_000:
            audio_np = self._resample(audio_np, self._sample_rate, 16_000)

        return self._transcribe(audio_np)

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Linear interpolation resampling using numpy."""
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        return np.interp(
            np.linspace(0, duration, target_len, endpoint=False),
            np.linspace(0, duration, len(audio), endpoint=False),
            audio
        ).astype(np.float32)

    # ── Recording ─────────────────────────────────────────────────────

    def _record_utterance(self) -> list:
        """
        Record audio chunks until silence follows speech, or the hard cap
        (_STT_MAX_SECONDS) is reached.  Returns a list of numpy arrays.
        """
        chunks:          list[np.ndarray] = []
        speech_detected: bool             = False
        silence_frames:  int              = 0
        total_frames:    int              = 0

        # silence_limit: how many consecutive silent chunks == end of utterance
        silence_limit = int(
            _STT_SILENCE_SECONDS * self._sample_rate / self._chunk_frames
        )
        max_frames = _STT_MAX_SECONDS * self._sample_rate

        print("[STT] Listening … (speak now)", flush=True)
        if self.display:
            self.display.set_state("listening")

        with sd.InputStream(
            device=self._device,
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._chunk_frames,
        ) as stream:
            while total_frames < max_frames:
                data, _ = stream.read(self._chunk_frames)
                total_frames += len(data)
                rms = float(np.sqrt(np.mean(data ** 2)))

                if rms >= _STT_ENERGY_THRESHOLD:
                    speech_detected = True
                    silence_frames  = 0
                    chunks.append(data.copy())
                elif speech_detected:
                    # Collect silence frames too so Whisper has trailing context
                    chunks.append(data.copy())
                    silence_frames += 1
                    if silence_frames >= silence_limit:
                        break
                # else: pre-speech silence — discard to save memory

        if not speech_detected:
            print("[STT] No speech detected.")
            if self.display:
                self.display.set_state("idle")
            return []

        print("[STT] Processing …", flush=True)
        if self.display:
            self.display.set_state("processing")
        return chunks

    # ── Transcription ─────────────────────────────────────────────────

    def _transcribe(self, audio: "np.ndarray") -> str | None:
        """Run Faster-Whisper on a float32 numpy array; return cleaned text."""
        segments, info = self._model.transcribe(
            audio,
            language="en",          # hardcode to English; remove for auto-detect
            beam_size=1,            # faster on CPU; raise to 5 for accuracy
            vad_filter=True,        # Whisper-side VAD as a second pass
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
        )

        text = " ".join(seg.text for seg in segments).strip()

        # Strip common Whisper hallucination artefacts on silence
        _HALLUCINATION_RE = re.compile(
            r"^\s*(?:thank you\.?|thanks\.?|you\.?|\.+|,+)\s*$",
            re.IGNORECASE,
        )
        if not text or _HALLUCINATION_RE.match(text):
            print("[STT] Nothing heard clearly.")
            return None

        return text


# ─────────────────────────────────────────────
# Text-to-Speech  (Piper)
# ─────────────────────────────────────────────

_piper_voice = None  # lazy-loaded singleton

def _load_piper_voice():
    """Load the Piper voice model once and cache it."""
    global _piper_voice
    if _piper_voice is not None:
        return _piper_voice
    if not _PIPER_AVAILABLE:
        return None
    if not DEFAULT_VOICE.exists():
        print(f"[TTS] Voice model not found at {DEFAULT_VOICE}. TTS disabled.")
        return None
    try:
        _piper_voice = PiperVoice.load(str(DEFAULT_VOICE))
        return _piper_voice
    except Exception as e:
        print(f"[TTS] Failed to load voice: {e}")
        return None


def speak(text: str):
    """Synthesise text and play it through the default audio output."""
    voice = _load_piper_voice()
    if voice is None:
        return
    tmp_path = None
    try:
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        with wave.open(tmp_path, "wb") as wav_file:
            if hasattr(voice, "synthesize_wav"):
                voice.synthesize_wav(text, wav_file)
            else:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(voice.config.sample_rate)
                voice.synthesize(text, wav_file)

        players = []
        if sys.platform == "darwin":
            p = shutil.which("afplay")
            if p: players.append(p)
        else:
            for p_name in ["pw-play", "paplay", "aplay", "play"]:
                p = shutil.which(p_name)
                if p: players.append(p)

        if not players:
            raise FileNotFoundError("No audio player found (tried afplay/pw-play/paplay/aplay/play).")

        last_err = None
        for player in players:
            cmd = [player, tmp_path]
            if os.path.basename(player) == "aplay":
                device = os.getenv("NEODOC_AUDIO_DEVICE")
                cmd = [player, "-q"]
                if device:
                    cmd.extend(["-D", device])
                cmd.append(tmp_path)

            try:
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e

        if last_err is not None:
            raise last_err
    except Exception as e:
        print(f"[TTS] Playback error: {e}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


class SpeechQueue:
    """Background speech queue so TTS doesn't block response streaming."""

    def __init__(self, enabled: bool, display=None):
        self.enabled = enabled
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self.display = display

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def say(self, text: str) -> None:
        if not self.enabled:
            return
        # Strip technical tags in brackets, e.g. [LOGGED], [RECORDS], [ID: 123]
        cleaned = re.sub(r'\[.*?\]', '', text or "").strip()
        if not cleaned:
            return
        self._queue.put(cleaned)

    def close(self) -> None:
        if not self.enabled:
            return
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=10)

    def wait(self) -> None:
        """Wait for all pending speech items to be processed."""
        if not self.enabled:
            return
        self._queue.join()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            try:
                if self.display:
                    self.display.set_state("speaking")
                speak(item)
            finally:
                self._queue.task_done()
                if self.display and self._queue.empty():
                    self.display.set_state("idle")


_STREAM_SPEAK_BOUNDARY_RE = re.compile(r"(?:(?<=[.!?])\s+)|\n+")
_PATIENT_BULLET_RE = re.compile(r"^\s*•\s*(.+?)\s+at\s+", re.MULTILINE)


def _drain_speakable_segments(buffer: str) -> tuple[list[str], str]:
    """Split a growing text buffer into speakable segments + remainder."""
    segments: list[str] = []
    while True:
        match = _STREAM_SPEAK_BOUNDARY_RE.search(buffer)
        if not match:
            break
        end = match.end()
        segment = buffer[:end].strip()
        if segment:
            segments.append(segment)
        buffer = buffer[end:]
    return segments, buffer


def _user_is_asking_for_patient_list(user_input: str) -> bool:
    text = (user_input or "").lower()
    return bool(re.search(r"\b(who|names?|list|patients?)\b", text))


def _reply_from_get_patients_tool_result(result: str) -> str:
    if not result:
        return "[RECORDS] No patient records found."
    if "No patients found" in result:
        return result.replace("[RECORDS] ", "").strip()
    names = [n.strip() for n in _PATIENT_BULLET_RE.findall(result) if n.strip()]
    if not names:
        return result.replace("[RECORDS] ", "").strip()
    if len(names) == 1:
        return f"You saw {names[0]} today."
    return "You saw these patients today: " + ", ".join(names) + "."


# ─────────────────────────────────────────────
# Database Helpers
# ─────────────────────────────────────────────

DB_FILE = DATA_DIR / "neodoc.db"

def init_db():
    """Initialize SQLite database and create tables if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patient_logs (
                id TEXT PRIMARY KEY,
                patient_name TEXT,
                note TEXT,
                time TEXT,
                date TEXT,
                logged_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id TEXT PRIMARY KEY,
                message TEXT,
                delay_minutes INTEGER,
                created_at TEXT,
                trigger_at TEXT,
                completed INTEGER
            )
        """)


_SHIFT_HOURS = {
    "morning":   (6,  12),
    "afternoon": (12, 18),
    "evening":   (18, 24),
    "all":       (0,  24),
}


def resolve_date(date_arg: str | None) -> str:
    now = datetime.now()
    if not date_arg or date_arg.lower() == "today":
        return now.strftime("%Y-%m-%d")
    if date_arg.lower() == "yesterday":
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        datetime.strptime(date_arg, "%Y-%m-%d")
        return date_arg
    except ValueError:
        return now.strftime("%Y-%m-%d")


# ─────────────────────────────────────────────
# Approval Helpers
# ─────────────────────────────────────────────

_APPROVAL_PENDING = "__APPROVAL_PENDING__"

def _confirm_text(prompt: str) -> bool:
    """Ask the user y/n in text mode. Returns True if approved."""
    while True:
        try:
            ans = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("  Please type 'y' to confirm or 'n' to cancel.")


def _confirm_voice(
    listener: "SpeechListener | None",
    speech: "SpeechQueue | None" = None,
    record: dict | None = None,
) -> bool:
    """
    Speak the patient details and approval prompt aloud, wait for TTS
    to finish, then listen for a voice 'yes' or 'no'.
    Falls back to text input if no listener is available.
    """
    if listener is None:
        return _confirm_text("  Approve? (y/n): ")

    # Build a natural-language spoken summary of what was heard
    if record:
        spoken_prompt = (
            f"I heard: Patient {record['patient_name']}. "
            f"Note: {record['note']}. "
            f"Time: {record['time']}. "
            f"Say yes to save this entry, or no to cancel."
        )
    else:
        spoken_prompt = "Please say yes to save this entry, or no to cancel."

    print(f"  [TTS PROMPT] {spoken_prompt}")
    print("  [APPROVAL] Say 'yes' to confirm or 'no' to cancel …")

    # Speak the full prompt and wait for playback to complete before
    # opening the microphone — avoids the mic picking up TTS audio.
    if speech is not None and speech.enabled:
        speech.say(spoken_prompt)
        speech.wait()   # blocks until the audio has finished playing

    for attempt in range(3):
        heard = listener.listen()
        if heard:
            low = heard.lower().strip(" .")
            if any(w in low for w in ("yes", "confirm", "correct", "approved", "log it", "save")):
                if speech is not None and speech.enabled:
                    speech.say("Confirmed. Saving entry.")
                    speech.wait()
                return True
            if any(w in low for w in ("no", "cancel", "wrong", "discard", "don't", "stop")):
                if speech is not None and speech.enabled:
                    speech.say("Cancelled. The entry will not be saved.")
                    speech.wait()
                return False
            # Re-prompt on ambiguous input
            retry_msg = f"I heard '{heard}'. Please say yes or no."
            print(f"  [APPROVAL] Heard '{heard}' — say 'yes' or 'no'.")
            if speech is not None and speech.enabled:
                speech.say(retry_msg)
                speech.wait()
        else:
            if attempt < 2:
                retry_msg = "I didn't catch that. Say yes to save, or no to cancel."
                print("  [APPROVAL] Nothing heard — please try again.")
                if speech is not None and speech.enabled:
                    speech.say(retry_msg)
                    speech.wait()

    print("  [APPROVAL] No clear answer heard — entry cancelled.")
    if speech is not None and speech.enabled:
        speech.say("No response received. The entry has been cancelled.")
        speech.wait()
    return False


# ─────────────────────────────────────────────
# Tool Executors
# ─────────────────────────────────────────────

def execute_tool(tool_name: str, tool_args: dict,
                 approve_fn=None) -> str:
    """
    Execute a tool and return its result string.
    For write operations (log_patient) the optional approve_fn(record) -> bool
    is called before committing to the database.  If it returns False the entry
    is discarded and a cancellation message is returned.
    """
    now = datetime.now()

    if tool_name == "log_patient":
        visit_time = tool_args.get("time", now.strftime("%H:%M"))
        record = {
            "id":           str(uuid.uuid4()),
            "patient_name": tool_args["patient_name"],
            "note":         tool_args["note"],
            "time":         visit_time,
            "date":         now.strftime("%Y-%m-%d"),
            "logged_at":    now.isoformat()
        }

        # ── Approval gate ─────────────────────────────────────────────
        print("\n  ┌─ APPROVAL REQUIRED ─────────────────────────────────┐")
        print(f"  │  Patient : {record['patient_name']}")
        print(f"  │  Note    : {record['note']}")
        print(f"  │  Time    : {record['time']}  |  Date: {record['date']}")
        print("  └─────────────────────────────────────────────────────┘")

        approved = approve_fn(record) if approve_fn else True

        if not approved:
            return (
                f"[CANCELLED] Entry for '{record['patient_name']}' was NOT saved. "
                "Please repeat your note if the details were incorrect."
            )

        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(
                "INSERT INTO patient_logs (id, patient_name, note, time, date, logged_at) VALUES (?, ?, ?, ?, ?, ?)",
                (record["id"], record["patient_name"], record["note"],
                 record["time"], record["date"], record["logged_at"])
            )
        return (
            f"[LOGGED] Patient '{record['patient_name']}' saved at {record['time']} "
            f"on {record['date']}. Note: {record['note']} (id: {record['id']})"
        )

    elif tool_name == "get_patients_today":
        shift = tool_args.get("shift", "all")
        query_date = resolve_date(tool_args.get("date"))
        start_h, end_h = _SHIFT_HOURS.get(shift, (0, 24))

        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM patient_logs WHERE date = ?", (query_date,)).fetchall()

        filtered = []
        for row in rows:
            entry = dict(row)
            try:
                hour = datetime.strptime(entry["time"], "%H:%M").hour
            except ValueError:
                hour = -1
            if shift == "all" or (start_h <= hour < end_h):
                filtered.append(entry)

        if not filtered:
            return f"[RECORDS] No patients found for shift '{shift}' on {query_date}."

        lines = [f"[RECORDS] {len(filtered)} patient(s) for shift '{shift}' on {query_date}:"]
        for e in filtered:
            lines.append(f"  • {e['patient_name']} at {e['time']} — {e['note']}")
        return "\n".join(lines)

    elif tool_name == "set_reminder":
        delay = int(tool_args["delay_minutes"])
        trigger_at = now + timedelta(minutes=delay)
        record = {
            "id":            str(uuid.uuid4()),
            "message":       tool_args["message"],
            "delay_minutes": delay,
            "created_at":    now.isoformat(),
            "trigger_at":    trigger_at.isoformat(),
            "completed":     0
        }
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(
                "INSERT INTO reminders (id, message, delay_minutes, created_at, trigger_at, completed) VALUES (?, ?, ?, ?, ?, ?)",
                (record["id"], record["message"], record["delay_minutes"], record["created_at"], record["trigger_at"], record["completed"])
            )
        return (
            f"[REMINDER SET] '{record['message']}' in {delay} min "
            f"(triggers at {trigger_at.strftime('%H:%M')}). id: {record['id']}"
        )

    elif tool_name == "get_patient_notes":
        patient_name = tool_args["patient_name"]
        date_filter  = tool_args.get("date")

        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            if date_filter:
                query_date = resolve_date(date_filter)
                rows = conn.execute(
                    "SELECT * FROM patient_logs WHERE LOWER(patient_name) LIKE ? AND date = ? ORDER BY logged_at",
                    (f"%{patient_name.lower()}%", query_date)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM patient_logs WHERE LOWER(patient_name) LIKE ? ORDER BY date DESC, logged_at DESC",
                    (f"%{patient_name.lower()}%",)
                ).fetchall()

        entries = [dict(r) for r in rows]
        if not entries:
            scope = f" on {resolve_date(date_filter)}" if date_filter else ""
            return f"[NOTES] No records found for patient matching '{patient_name}'{scope}."

        lines = [f"[NOTES] {len(entries)} record(s) for '{patient_name}':"]
        for e in entries:
            lines.append(
                f"  • [{e['date']} {e['time']}] {e['patient_name']} — {e['note']}"
                f" (id: {e['id']})"
            )
        return "\n".join(lines)

    elif tool_name == "update_patient_note":
        patient_name = tool_args["patient_name"]
        new_note     = tool_args["new_note"]
        query_date   = resolve_date(tool_args.get("date"))

        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM patient_logs WHERE LOWER(patient_name) LIKE ? AND date = ? ORDER BY logged_at DESC LIMIT 1",
                (f"%{patient_name.lower()}%", query_date)
            ).fetchone()
            if not row:
                return f"[UPDATE FAILED] No log entry found for '{patient_name}' on {query_date}."
            entry = dict(row)
            conn.execute(
                "UPDATE patient_logs SET note = ? WHERE id = ?",
                (new_note, entry["id"])
            )
        return (
            f"[UPDATED] Note for '{entry['patient_name']}' on {query_date} at {entry['time']} "
            f"updated to: '{new_note}' (id: {entry['id']})"
        )

    elif tool_name == "delete_patient_log":
        entry_id = tool_args["entry_id"]
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM patient_logs WHERE id = ?", (entry_id,)
            ).fetchone()
            if not row:
                return f"[DELETE FAILED] No log entry found with id '{entry_id}'."
            entry = dict(row)
            conn.execute("DELETE FROM patient_logs WHERE id = ?", (entry_id,))
        return (
            f"[DELETED] Removed log for '{entry['patient_name']}' from {entry['date']} at {entry['time']}. "
            f"Note was: '{entry['note']}' (id: {entry_id})"
        )

    elif tool_name == "summarise_shift":
        fmt = tool_args.get("format", "brief")
        query_date = resolve_date(tool_args.get("date"))

        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM patient_logs WHERE date = ? ORDER BY time", (query_date,)
            ).fetchall()

        day_logs = [dict(row) for row in rows]

        if not day_logs:
            return f"[SHIFT SUMMARY] No patient records found for {query_date}."

        header = f"[SHIFT SUMMARY — {fmt.upper()}] {query_date} | {len(day_logs)} patient(s):"
        if fmt == "brief":
            names = ", ".join(e["patient_name"] for e in day_logs)
            return f"{header} {names}"
        elif fmt == "handover":
            lines = [header, "", "  HANDOVER NOTES:", "  " + "─" * 40]
            for e in day_logs:
                lines.append(f"  [{e['time']}] {e['patient_name']}")
                lines.append(f"    Note: {e['note']}")
                lines.append(f"    Entry ID: {e['id']}")
                lines.append("")
            return "\n".join(lines)
        else:  # detailed
            lines = [header]
            for e in day_logs:
                lines.append(f"  • [{e['time']}] {e['patient_name']} — {e['note']} (id: {e['id']})")
            return "\n".join(lines)

    else:
        return f"[ERROR] Unknown tool: {tool_name}"


# ─────────────────────────────────────────────
# Ollama API
# ─────────────────────────────────────────────

def call_ollama(
    messages: list,
    model: str,
    *,
    stream: bool = False,
    on_chunk=None,
) -> tuple[str, list]:
    """Send messages to Ollama; returns (content, tool_calls)."""
    try:
        if not stream:
            response = ollama.chat(
                model=model,
                messages=messages,
                tools=TOOLS,
            )
            msg = response.message
            return (msg.content or ""), (msg.tool_calls or [])

        content_parts: list[str] = []
        tool_calls: list = []
        for part in ollama.chat(
            model=model,
            messages=messages,
            tools=TOOLS,
            stream=True,
        ):
            msg = part.message
            if msg.content:
                content_parts.append(msg.content)
                if on_chunk is not None:
                    on_chunk(msg.content)
            if msg.tool_calls:
                tool_calls = list(msg.tool_calls)

        return "".join(content_parts), tool_calls

    except ollama.ResponseError as e:
        print(f"\n[ERROR] Ollama returned an error: {e.error}")
        sys.exit(1)
    except Exception as e:
        if "connection" in str(e).lower() or "refused" in str(e).lower():
            print("\n[ERROR] Cannot connect to Ollama. Is it running?")
            print("  Start it with:  ollama serve")
        else:
            print(f"\n[ERROR] Unexpected error communicating with Ollama: {e}")
        sys.exit(1)


def run_turn(
    user_input: str,
    history: list,
    model: str,
    *,
    stream: bool = False,
    speech: "SpeechQueue | None" = None,
    listener: "SpeechListener | None" = None,
) -> tuple[str, list, dict]:
    """
    Run one full turn: user prompt → model → tool call(s) → final reply.
    Returns (final_reply, updated_history, turn_log_dict).
    """
    history.append({"role": "user", "content": user_input})

    turn_log = {
        "timestamp":   datetime.now().isoformat(),
        "user_input":  user_input,
        "tool_calls":  [],
        "final_reply": ""
    }

    # Build approval callback — used by execute_tool before writing log_patient
    def _approve_fn(_record: dict) -> bool:
        if listener:
            return _confirm_voice(listener, speech=speech, record=_record)
        return _confirm_text("  Approve this entry? (y/n): ")

    # ── First model call ──────────────────────────────────────────────
    first_content = ""
    first_tool_calls: list = []

    if stream:
        streamed_any = False
        tts_buffer = ""

        def on_first_chunk(delta: str):
            nonlocal streamed_any, tts_buffer
            if not streamed_any:
                print("\nAssistant: ", end="", flush=True)
                streamed_any = True
            print(delta, end="", flush=True)
            if speech is not None and speech.enabled:
                tts_buffer += delta
                segments, remainder = _drain_speakable_segments(tts_buffer)
                tts_buffer = remainder
                for seg in segments:
                    speech.say(seg)

        first_content, first_tool_calls = call_ollama(history, model, stream=True, on_chunk=on_first_chunk)
        if streamed_any:
            if speech is not None and speech.enabled and tts_buffer.strip():
                speech.say(tts_buffer.strip())
            print("\n")
    else:
        first_content, first_tool_calls = call_ollama(history, model, stream=False)

    tool_calls = first_tool_calls

    if not tool_calls:
        reply = first_content or "[No response]"
        history.append({"role": "assistant", "content": reply})
        turn_log["final_reply"] = reply
        if (not stream) and (speech is not None) and speech.enabled:
            speech.say(reply)
        return reply, history, turn_log

    # ── Tool execution loop ───────────────────────────────────────────
    history.append({"role": "assistant", "content": first_content or "", "tool_calls": tool_calls})

    tool_results_by_name: dict[str, list[str]] = {}

    for tc in tool_calls:
        tool_name = tc.function.name
        tool_args = tc.function.arguments or {}
        result    = execute_tool(tool_name, tool_args, approve_fn=_approve_fn)

        tool_results_by_name.setdefault(tool_name, []).append(result)
        turn_log["tool_calls"].append({"tool": tool_name, "args": tool_args, "result": result})

        history.append({"role": "tool", "content": result, "tool_name": tool_name})

        # Don't speak tool results directly; wait for the final assistant response.
        # (Removed speech.say(result) here to avoid reading technical logs)

        print(f"\n  ⚙  {tool_name}  args={tool_args}")
        print(f"     → {result}")

    # Check if any tool result was cancelled (model might struggle to reply to these)
    cancelled_results = [r["result"] for r in turn_log["tool_calls"] if "[CANCELLED]" in r["result"]]
    if cancelled_results:
        # Clean up the technical tag for the final reply
        reply = cancelled_results[-1].replace("[CANCELLED]", "").strip()
        history.append({"role": "assistant", "content": reply})
        turn_log["final_reply"] = reply
        if stream:
            print(f"\n\nAssistant: {reply}\n")
        else:
            print(f"\nAssistant: {reply}\n")
        if speech is not None and speech.enabled:
            speech.say(reply)
        return reply, history, turn_log

    # ── Second model call ─────────────────────────────────────────────
    if any(tc.function.name == "get_patients_today" for tc in tool_calls) and _user_is_asking_for_patient_list(user_input):
        results = tool_results_by_name.get("get_patients_today", [])
        reply = _reply_from_get_patients_tool_result(results[-1] if results else "")

        if stream:
            print("\n\nAssistant: " + reply + "\n")
        if not stream:
            print(f"\nAssistant: {reply}\n")
        if (speech is not None) and speech.enabled:
            speech.say(reply)

        history.append({"role": "assistant", "content": reply})
        turn_log["final_reply"] = reply
        return reply, history, turn_log

    if stream:
        print("\n\nAssistant: ", end="", flush=True)
        tts_buffer = ""

        def on_second_chunk(delta: str):
            nonlocal tts_buffer
            print(delta, end="", flush=True)
            if speech is not None and speech.enabled:
                tts_buffer += delta
                segments, remainder = _drain_speakable_segments(tts_buffer)
                tts_buffer = remainder
                for seg in segments:
                    speech.say(seg)

        reply, _ = call_ollama(history, model, stream=True, on_chunk=on_second_chunk)
        if (speech is not None) and speech.enabled and tts_buffer.strip():
            speech.say(tts_buffer.strip())
        print("\n")
    else:
        reply, _ = call_ollama(history, model, stream=False)
        print(f"\nAssistant: {reply}\n")
        if (speech is not None) and speech.enabled:
            speech.say(reply)

    reply = reply or "[No response]"
    history.append({"role": "assistant", "content": reply})
    turn_log["final_reply"] = reply
    return reply, history, turn_log


# ─────────────────────────────────────────────
# Session Logging
# ─────────────────────────────────────────────

def save_session(session_log: dict):
    """Save session to JSON and a plain-text summary."""
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = LOGS_DIR / f"session_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(session_log, f, indent=2)

    txt_path = LOGS_DIR / f"session_{timestamp}.txt"
    with open(txt_path, "w") as f:
        f.write(f"Session: {session_log['session_start']}\n")
        f.write(f"Model:   {session_log['model']}\n")
        f.write(f"Turns:   {len(session_log['turns'])}\n")
        f.write("=" * 60 + "\n\n")
        for i, turn in enumerate(session_log["turns"], 1):
            f.write(f"[Turn {i}] {turn['timestamp']}\n")
            f.write(f"  You:       {turn['user_input']}\n")
            if turn["tool_calls"]:
                for tc in turn["tool_calls"]:
                    f.write(f"  Tool:      {tc['tool']}({tc['args']})\n")
                    f.write(f"  Result:    {tc['result']}\n")
            f.write(f"  Assistant: {turn['final_reply']}\n\n")

    return json_path, txt_path


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

def list_tools():
    width = 60
    print(f"\n{'─' * width}")
    print(f"  Available Clinical Tools  ({len(TOOLS)} total)")
    print(f"{'─' * width}\n")
    for t in TOOLS:
        fn       = t["function"]
        props    = fn["parameters"]["properties"]
        required = fn["parameters"].get("required", [])
        print(f"  \033[1m{fn['name']}\033[0m")
        print(f"  {fn['description']}")
        if props:
            print(f"  Parameters:")
            for pname, pdef in props.items():
                req_flag = " [required]" if pname in required else " [optional]"
                ptype    = pdef.get("type", "string")
                pdesc    = pdef.get("description", "")
                enum_str = ""
                if "enum" in pdef:
                    enum_str = f"  choices: {pdef['enum']}"
                print(f"    • {pname} ({ptype}){req_flag}")
                print(f"      {pdesc}{enum_str}")
        print()
    print(f"{'─' * width}\n")



def list_audio_devices():
    """Print available audio input/output devices."""
    if not _SOUNDDEVICE_AVAILABLE:
        print("[ERROR] sounddevice not installed.")
        return
    print("\nAvailable Audio Devices:")
    print(sd.query_devices())
    print(f"\nDefault Input Device : {sd.default.device[0]}")
    print(f"Default Output Device: {sd.default.device[1]}\n")


def _get_user_input_voice(listener: "SpeechListener") -> str | None:
    """
    Attempt to capture and transcribe one voice utterance.
    Returns the transcribed text, or None if nothing was heard.
    Prints the heard text so the user can confirm what was understood.
    """
    text = listener.listen()
    if text:
        print(f"You (heard): {text}")
    return text


def main():
    parser = argparse.ArgumentParser(description="Clinical AI Tool Call Tester")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--list-tools", action="store_true",
                        help="Print available tools and exit")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable text-to-speech output")
    parser.add_argument("--voice", action="store_true",
                        help="Enable voice input via microphone (Faster-Whisper STT)")
    parser.add_argument("--stt-model", default=DEFAULT_STT,
                        metavar="SIZE",
                        help=f"Whisper model size for STT: tiny|base|small (default: {DEFAULT_STT})")
    parser.add_argument("--mic-device",
                        help="Input device ID or name for microphone (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true",
                        help="Show available audio devices and exit")
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream assistant output (default: enabled)",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    if args.list_tools:
        list_tools()
        sys.exit(0)

    init_db()

    # ── Display setup ─────────────────────────────────────────────────
    display = None
    if _OLED_AVAILABLE:
        display = DisplayController()
        display.start()

    # ── TTS setup ─────────────────────────────────────────────────────
    tts_enabled = not args.no_tts and _PIPER_AVAILABLE and DEFAULT_VOICE.exists()
    speech = SpeechQueue(enabled=tts_enabled, display=display)
    speech.start()

    # ── STT setup ─────────────────────────────────────────────────────
    listener: SpeechListener | None = None
    if args.voice:
        if not _WHISPER_AVAILABLE or not _SOUNDDEVICE_AVAILABLE:
            print("[ERROR] Voice input requires faster-whisper and sounddevice.")
            print("  pip install faster-whisper sounddevice numpy --break-system-packages")
            sys.exit(1)
        mic_device = args.mic_device or os.getenv("NEODOC_STT_DEVICE")
        try:
            # Try to parse as integer if possible
            mic_device = int(mic_device)
        except (ValueError, TypeError):
            pass

        listener = SpeechListener(model_size=args.stt_model, device=mic_device, display=display)

    # ── Banner ────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  Clinical AI Tool Tester")
    print(f"  Model : {args.model}")
    print(f"  Logs  : ./{LOGS_DIR}/")
    print(f"  TTS   : {'enabled (Piper)' if tts_enabled else 'disabled'}")
    print(f"  STT   : {'enabled (Whisper ' + args.stt_model + ')' if listener else 'disabled (text mode)'}")
    print(f"{'─'*55}")
    if listener:
        print("  Speak after the '[STT] Listening …' prompt.")
        print("  Say 'quit' or 'exit' to end the session.")
    else:
        print("  Type your prompt and press Enter.")
        print("  Commands: 'quit' to exit | 'clear' to reset history")
    print(f"{'─'*55}\n")

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    session_log = {
        "session_start": datetime.now().isoformat(),
        "model":         args.model,
        "turns":         []
    }

    try:
        while True:
            # ── Get input: voice or keyboard ──────────────────────────
            if listener:
                user_input = _get_user_input_voice(listener)
                if user_input is None:
                    # Nothing heard — loop back and listen again
                    continue
            else:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

            if not user_input:
                continue

            # Shared exit / control commands (work in both text and voice mode)
            cmd = user_input.lower().strip(" .")
            if cmd in ("quit", "exit", "goodbye", "bye"):
                break
            if cmd == "clear":
                history = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("[Conversation history cleared]\n")
                continue

            print("loading...", end="\r")
            if display:
                display.set_state("processing")
            reply, history, turn_log = run_turn(
                user_input,
                history,
                args.model,
                stream=args.stream,
                speech=speech,
                listener=listener,
            )
            session_log["turns"].append(turn_log)

            # ── Wait for speech to finish before next loop iteration ──────
            # This prevents the microphone from picking up the assistant's own voice.
            if speech:
                speech.wait()

            if display:
                display.set_state("idle")

            if not args.stream and not turn_log["tool_calls"]:
                print(f"\nAssistant: {reply}\n")

    finally:
        if display:
            try:
                display.stop()
            except Exception:
                pass
        try:
            speech.close()
        except KeyboardInterrupt:
            pass
        if session_log["turns"]:
            json_path, txt_path = save_session(session_log)
            print(f"\nSession saved:")
            print(f"  JSON → {json_path}")
            print(f"  TXT  → {txt_path}")
        else:
            print("\nNo turns recorded. Nothing saved.")


if __name__ == "__main__":
    main()