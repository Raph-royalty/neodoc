"""
Clinical AI Assistant — Tool Call Tester
-----------------------------------------
Tests tool calling with a local Ollama model via text prompts.
Results are saved to session logs (JSON + plain text summary).
Tool output (patient logs, reminders) is persisted to JSON files under data/.

Usage:
    python main.py                         # Default model
    python main.py --model granite3.2:3b   # Different model
    python main.py --list-tools            # Show available tools
"""

import argparse
import json
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import ollama

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────


DEFAULT_MODEL = "qwen3.5:0.8b"
LOGS_DIR = Path("logs")
DATA_DIR = Path("data")
TOOLS_FILE = Path("tools.json")

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
# JSON Database Helpers
# ─────────────────────────────────────────────

PATIENT_LOGS_FILE = DATA_DIR / "patient_logs.json"
REMINDERS_FILE    = DATA_DIR / "reminders.json"


def _read_json(path: Path) -> list:
    """Read a JSON array from a file, returning [] if missing or empty."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, data: list):
    """Write a JSON array to a file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _append_json(path: Path, record: dict):
    """Append a single record to a JSON array file."""
    records = _read_json(path)
    records.append(record)
    _write_json(path, records)


# Shift time boundaries (24-h)
_SHIFT_HOURS = {
    "morning":   (6,  12),
    "afternoon": (12, 18),
    "evening":   (18, 24),
    "all":       (0,  24),
}


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
        _append_json(PATIENT_LOGS_FILE, record)
        return (
            f"[LOGGED] Patient '{record['patient_name']}' saved at {visit_time} "
            f"on {record['date']}. Note: {record['note']} (id: {record['id']})"
        )

    # ── get_patients_today ────────────────────────────────────────────
    elif tool_name == "get_patients_today":
        shift = tool_args.get("shift", "all")
        today = now.strftime("%Y-%m-%d")
        start_h, end_h = _SHIFT_HOURS.get(shift, (0, 24))

        all_logs = _read_json(PATIENT_LOGS_FILE)
        filtered = []
        for entry in all_logs:
            if entry.get("date") != today:
                continue
            # Parse hour from stored time ("HH:MM" or freeform like "3pm")
            try:
                hour = datetime.strptime(entry["time"], "%H:%M").hour
            except ValueError:
                hour = -1  # keep freeform entries under "all"
            if shift == "all" or (start_h <= hour < end_h):
                filtered.append(entry)

        if not filtered:
            return f"[RECORDS] No patients found for shift '{shift}' on {today}."

        lines = [f"[RECORDS] {len(filtered)} patient(s) for shift '{shift}' on {today}:"]
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
            "completed":    False
        }
        _append_json(REMINDERS_FILE, record)
        return (
            f"[REMINDER SET] '{record['message']}' in {delay} min "
            f"(triggers at {trigger_at.strftime('%H:%M')}). id: {record['id']}"
        )

    # ── query_drug_info ───────────────────────────────────────────────
    elif tool_name == "query_drug_info":
        # TODO: replace with local drug reference DB / RAG
        context = tool_args.get("patient_context", "adult")
        return (
            f"[DRUG INFO — STUB] {tool_args['drug_name'].capitalize()} / "
            f"{tool_args['query_type']} / context: {context}. "
            "Drug reference DB not yet implemented — use clinical guidelines."
        )

    # ── summarise_shift ───────────────────────────────────────────────
    elif tool_name == "summarise_shift":
        fmt = tool_args.get("format", "brief")
        today = now.strftime("%Y-%m-%d")
        all_logs = _read_json(PATIENT_LOGS_FILE)
        today_logs = [e for e in all_logs if e.get("date") == today]

        if not today_logs:
            return f"[SHIFT SUMMARY] No patient records found for today ({today})."

        header = f"[SHIFT SUMMARY — {fmt.upper()}] {today} | {len(today_logs)} patient(s):"
        if fmt == "brief":
            names = ", ".join(e["patient_name"] for e in today_logs)
            return f"{header} {names}"
        else:
            lines = [header]
            for e in today_logs:
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
            think=False,        # disable chain-of-thought / thinking tokens
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
    args = parser.parse_args()

    if args.list_tools:
        list_tools()
        sys.exit(0)

    print(f"\n{'─'*55}")
    print(f"  Clinical AI Tool Tester")
    print(f"  Model : {args.model}")
    print(f"  Logs  : ./{LOGS_DIR}/")
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

            print("Thinking...", end="\r")
            reply, history, turn_log = run_turn(user_input, history, args.model)
            session_log["turns"].append(turn_log)

            # Print tool calls if any
            if turn_log["tool_calls"]:
                for tc in turn_log["tool_calls"]:
                    print(f"\n  ⚙  {tc['tool']}  args={tc['args']}")
                    print(f"     → {tc['result']}")

            print(f"\nAssistant: {reply}\n")

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