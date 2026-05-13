"""
Clinical AI Assistant — Tool Call Tester
-----------------------------------------
Tests tool calling with a local Ollama model via text prompts.
Results are saved to session logs (JSON + plain text summary).
Tool output (patient logs, reminders) is persisted to JSON files under data/.

Usage:
    python main.py                         # Default model
    python main.py --model qwen3.5:0.8b   # Different model
    python main.py --list-tools            # Show available tools
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
import uuid
import wave
from datetime import datetime, timedelta
from pathlib import Path

import ollama

try:
    from piper.voice import PiperVoice
    _PIPER_AVAILABLE = True
except ImportError:
    _PIPER_AVAILABLE = False

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────


DEFAULT_MODEL = "granite4:350m"
LOGS_DIR = Path("logs")
DATA_DIR = Path("data")
TOOLS_FILE = Path("tools.json")
VOICES_DIR = Path("voices")
DEFAULT_VOICE = VOICES_DIR / "en_US-lessac-medium.onnx"

SYSTEM_PROMPT = """You are a clinical assistant helping nurses and doctors in a ward setting.
You have access to tools for logging patient visits, retrieving records,
setting reminders, querying drug information, and summarising shift notes.
Always use the appropriate tool when the user's request matches a tool's purpose.
Be concise and professional in your responses."""


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
# Text-to-Speech (Piper)
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
        # Write WAV to a temp file; audio players require a real file path
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        with wave.open(tmp_path, "wb") as wav_file:
            # Newer piper-tts uses synthesize_wav(); older versions used synthesize(text, wav_file)
            if hasattr(voice, "synthesize_wav"):
                voice.synthesize_wav(text, wav_file)
            else:
                wav_file.setnchannels(1)      # mono
                wav_file.setsampwidth(2)      # 16-bit PCM
                wav_file.setframerate(voice.config.sample_rate)
                voice.synthesize(text, wav_file)
        # Play synchronously then remove the temp file
        player = None
        if sys.platform == "darwin":
            player = shutil.which("afplay")
        else:
            # Linux: prefer the desktop audio stack (PipeWire/Pulse) before raw ALSA.
            # This avoids cases where aplay targets a silent/non-default ALSA device.
            player = (
                shutil.which("pw-play")
                or shutil.which("paplay")
                or shutil.which("aplay")
                or shutil.which("play")
            )

        if player is None:
            raise FileNotFoundError("No audio player found (tried afplay/aplay/paplay/play).")

        cmd = [player, tmp_path]
        if os.path.basename(player) == "aplay":
            # Optional ALSA device override, e.g. NEODOC_AUDIO_DEVICE=hw:0,0
            device = os.getenv("NEODOC_AUDIO_DEVICE")
            cmd = [player, "-q"]
            if device:
                cmd.extend(["-D", device])
            cmd.append(tmp_path)

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception as e:
        print(f"[TTS] Playback error: {e}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


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



# Shift time boundaries (24-h)
_SHIFT_HOURS = {
    "morning":   (6,  12),
    "afternoon": (12, 18),
    "evening":   (18, 24),
    "all":       (0,  24),
}


def resolve_date(date_arg: str | None) -> str:
    """
    Resolve a date argument to a YYYY-MM-DD string.
    Accepts: None / 'today' → today, 'yesterday' → yesterday,
             or any YYYY-MM-DD string (returned as-is).
    """
    now = datetime.now()
    if not date_arg or date_arg.lower() == "today":
        return now.strftime("%Y-%m-%d")
    if date_arg.lower() == "yesterday":
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")
    # Validate ISO format; fall back to today on parse error
    try:
        datetime.strptime(date_arg, "%Y-%m-%d")
        return date_arg
    except ValueError:
        return now.strftime("%Y-%m-%d")


# ─────────────────────────────────────────────
# Tool Executors  (JSON file-backed)
# ─────────────────────────────────────────────

def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    Executes a tool call, persists data to JSON files, and returns a result string.
    Swap the JSON helpers for SQLite/APScheduler when ready.
    """
    now = datetime.now()

    # ── log_patient ───────────────────────────────────────────────────
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
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(
                "INSERT INTO patient_logs (id, patient_name, note, time, date, logged_at) VALUES (?, ?, ?, ?, ?, ?)",
                (record["id"], record["patient_name"], record["note"], record["time"], record["date"], record["logged_at"])
            )
        return (
            f"[LOGGED] Patient '{record['patient_name']}' saved at {visit_time} "
            f"on {record['date']}. Note: {record['note']} (id: {record['id']})"
        )

    # ── get_patients_today ────────────────────────────────────────────
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
            # Parse hour from stored time ("HH:MM" or freeform like "3pm")
            try:
                hour = datetime.strptime(entry["time"], "%H:%M").hour
            except ValueError:
                hour = -1  # keep freeform entries under "all"
            if shift == "all" or (start_h <= hour < end_h):
                filtered.append(entry)

        if not filtered:
            return f"[RECORDS] No patients found for shift '{shift}' on {query_date}."

        lines = [f"[RECORDS] {len(filtered)} patient(s) for shift '{shift}' on {query_date}:"]
        for e in filtered:
            lines.append(f"  • {e['patient_name']} at {e['time']} — {e['note']}")
        return "\n".join(lines)

    # ── set_reminder ──────────────────────────────────────────────────
    elif tool_name == "set_reminder":
        delay = int(tool_args["delay_minutes"])
        trigger_at = now + timedelta(minutes=delay)
        record = {
            "id":           str(uuid.uuid4()),
            "message":      tool_args["message"],
            "delay_minutes": delay,
            "created_at":   now.isoformat(),
            "trigger_at":   trigger_at.isoformat(),
            "completed":    0
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

    # ── summarise_shift ───────────────────────────────────────────────
    elif tool_name == "summarise_shift":
        fmt = tool_args.get("format", "brief")
        query_date = resolve_date(tool_args.get("date"))

        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM patient_logs WHERE date = ?", (query_date,)).fetchall()

        day_logs = [dict(row) for row in rows]

        if not day_logs:
            return f"[SHIFT SUMMARY] No patient records found for {query_date}."

        header = f"[SHIFT SUMMARY — {fmt.upper()}] {query_date} | {len(day_logs)} patient(s):"
        if fmt == "brief":
            names = ", ".join(e["patient_name"] for e in day_logs)
            return f"{header} {names}"
        else:
            lines = [header]
            for e in day_logs:
                lines.append(f"  • [{e['time']}] {e['patient_name']} — {e['note']}")
            return "\n".join(lines)

    # ── unknown ───────────────────────────────────────────────────────
    else:
        return f"[ERROR] Unknown tool: {tool_name}"


# ─────────────────────────────────────────────
# Ollama API
# ─────────────────────────────────────────────

def call_ollama(messages: list, model: str) -> ollama.ChatResponse:
    """Send messages to Ollama using the ollama package and return the response."""
    try:
        return ollama.chat(
            model=model,
            messages=messages,
            tools=TOOLS,
        )
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


def run_turn(user_input: str, history: list, model: str) -> tuple[str, list, dict]:
    """
    Run one full turn: user prompt → model → tool call(s) → final reply.
    Returns (final_reply, updated_history, turn_log_dict).
    """
    history.append({"role": "user", "content": user_input})

    turn_log = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "tool_calls": [],
        "final_reply": ""
    }

    # ── First model call ──────────────────────────────────────────────
    response = call_ollama(history, model)
    message = response.message
    tool_calls = message.tool_calls or []

    if not tool_calls:
        # No tool needed — direct answer
        reply = message.content or "[No response]"
        history.append({"role": "assistant", "content": reply})
        turn_log["final_reply"] = reply
        return reply, history, turn_log

    # ── Tool execution loop ───────────────────────────────────────────
    history.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

    for tc in tool_calls:
        tool_name = tc.function.name
        tool_args = tc.function.arguments or {}

        result = execute_tool(tool_name, tool_args)

        # Log the tool call
        turn_log["tool_calls"].append({
            "tool": tool_name,
            "args": tool_args,
            "result": result
        })

        # Feed result back to model
        history.append({
            "role": "tool",
            "content": result
        })

    # ── Second model call — generate final reply using tool results ───
    response2 = call_ollama(history, model)
    reply = response2.message.content or "[No response]"
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

    # JSON log (full detail)
    json_path = LOGS_DIR / f"session_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(session_log, f, indent=2)

    # Plain text summary
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
    print("\nAvailable tools:\n")
    for t in TOOLS:
        fn = t["function"]
        params = ", ".join(fn["parameters"]["properties"].keys())
        print(f"  {fn['name']}({params})")
        print(f"    {fn['description']}\n")


def main():
    parser = argparse.ArgumentParser(description="Clinical AI Tool Call Tester")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--list-tools", action="store_true",
                        help="Print available tools and exit")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable text-to-speech output")
    args = parser.parse_args()

    if args.list_tools:
        list_tools()
        sys.exit(0)

    init_db()

    tts_enabled = not args.no_tts and _PIPER_AVAILABLE and DEFAULT_VOICE.exists()

    print(f"\n{'─'*55}")
    print(f"  Clinical AI Tool Tester")
    print(f"  Model : {args.model}")
    print(f"  Logs  : ./{LOGS_DIR}/")
    print(f"  TTS   : {'enabled (Piper)' if tts_enabled else 'disabled'}")
    print(f"{'─'*55}")
    print("  Type your prompt and press Enter.")
    print("  Commands: 'quit' to exit | 'clear' to reset history")
    print(f"{'─'*55}\n")

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    session_log = {
        "session_start": datetime.now().isoformat(),
        "model": args.model,
        "turns": []
    }

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                break
            if user_input.lower() == "clear":
                history = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("[Conversation history cleared]\n")
                continue

            print("loading...", end="\r")
            reply, history, turn_log = run_turn(user_input, history, args.model)
            session_log["turns"].append(turn_log)

            # Print tool calls if any
            if turn_log["tool_calls"]:
                for tc in turn_log["tool_calls"]:
                    print(f"\n  ⚙  {tc['tool']}  args={tc['args']}")
                    print(f"     → {tc['result']}")

            print(f"\nAssistant: {reply}\n")

            if tts_enabled:
                speak(reply)

    finally:
        if session_log["turns"]:
            json_path, txt_path = save_session(session_log)
            print(f"\nSession saved:")
            print(f"  JSON → {json_path}")
            print(f"  TXT  → {txt_path}")
        else:
            print("\nNo turns recorded. Nothing saved.")


if __name__ == "__main__":
    main()