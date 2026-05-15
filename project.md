# NeoDoc: A Voice-Driven Clinical AI Assistant
### Mini Project Report — Embedded Intelligent Systems

---

## Table of Contents

1. [Project Description](#1-project-description)
2. [Justification](#2-justification)
3. [System Architecture](#3-system-architecture)
4. [Implementation Details](#4-implementation-details)
   - [4.1 Main Application (`main.py`)](#41-main-application-mainpy)
   - [4.2 OLED Display Controller (`display.py`)](#42-oled-display-controller-displaypy)
   - [4.3 Tool System (`tools.json`)](#43-tool-system-toolsjson)
5. [AI Implementation](#5-ai-implementation)
   - [5.1 Language Model (LLM)](#51-language-model-llm)
   - [5.2 Speech-to-Text (STT)](#52-speech-to-text-stt)
   - [5.3 Text-to-Speech (TTS)](#53-text-to-speech-tts)
   - [5.4 Edge Deployment](#54-edge-deployment)
6. [Hardware Implementation](#6-hardware-implementation)
   - [6.1 Microcontroller / SBC](#61-microcontroller--sbc)
   - [6.2 OLED Display](#62-oled-display)
   - [6.3 Microphone Input](#63-microphone-input)
   - [6.4 Audio Output](#64-audio-output)
   - [6.5 Physical Button (GPIO)](#65-physical-button-gpio)
7. [Runtime Screenshots & Output](#7-runtime-screenshots--output)
8. [Testing Results](#8-testing-results)
9. [Conclusion](#9-conclusion)

---

## 1. Project Description

**NeoDoc** is a voice-driven clinical AI assistant designed to support nurses and doctors in active hospital ward environments. Built as a fully embedded, offline-capable intelligent system, it enables healthcare workers to interact naturally using spoken language to log patient visits, retrieve clinical notes, set reminders, and generate shift handover summaries — all without touching a keyboard or screen.

The system integrates three AI components on a single-board computer:

| Component | Technology |
|---|---|
| Large Language Model (LLM) | IBM Granite 4 (350M params) via Ollama |
| Speech-to-Text (STT) | Faster-Whisper (`base` model, `int8` quantised) |
| Text-to-Speech (TTS) | Piper TTS (`en_US-lessac-medium` voice) |

All AI inference runs **entirely on-device** — no cloud API calls, no internet dependency. Patient data is stored in a local SQLite database, and session logs are written as JSON files for audit traceability.

A secondary embedded component — a **128×64 OLED display** — provides real-time animated visual feedback of the assistant's internal state (idle, listening, processing, or speaking), giving ward staff an immediate, at-a-glance indication of system activity.

The system supports two interaction modes:
- **Voice mode** (`--voice`): hands-free microphone input with VAD (Voice Activity Detection)
- **Text mode** (default): keyboard prompts for testing and development

---

## 2. Justification

Clinical environments present unique challenges for digital documentation:

- **Hands are often occupied** — nurses may be performing procedures, wearing gloves, or managing equipment when a critical observation needs to be recorded.
- **Time pressure is extreme** — manually opening an application, typing a note, and navigating a UI wastes precious seconds that could affect patient outcomes.
- **Recall degradation** — notes written at the end of a shift from memory are less accurate than notes captured at the point of care.
- **Handover risk** — incomplete or ambiguous handover notes are a documented source of clinical error.

NeoDoc addresses all of these by enabling **point-of-care voice documentation**: the clinician speaks naturally, the system understands intent, confirms the entry, and persists the record — all within seconds.

Additionally, **data privacy** is a paramount concern in healthcare. Deploying AI at the edge (on-device) eliminates the need to transmit patient data to external servers, ensuring compliance with data protection regulations.

The use of **small, quantised language models** (350M–8B parameters) makes this feasible on affordable single-board hardware, enabling deployment in resource-constrained settings such as rural clinics or mobile field hospitals.

---

## 3. System Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                       NeoDoc System (On-Device)                       │
│                                                                       │
│  ┌──────────────┐    ┌──────────────────────────────────────────────┐ │
│  │   Hardware   │    │              main.py (Orchestrator)           │ │
│  │              │    │                                              │ │
│  │  Microphone  │───▶│  SpeechListener (Faster-Whisper STT)        │ │
│  │              │    │         │                                    │ │
│  │  OLED 128x64 │◀───│  DisplayController (display.py)             │ │
│  │              │    │         │                                    │ │
│  │  Speaker /   │◀───│  SpeechQueue (Piper TTS)                    │ │
│  │  Bluetooth   │    │         │                                    │ │
│  │              │    │         ▼                                    │ │
│  │  GPIO Button │───▶│  Ollama API (LLM — granite4:350m)           │ │
│  │              │    │         │                                    │ │
│  └──────────────┘    │         ▼                                    │ │
│                       │  Tool Executor (execute_tool)               │ │
│                       │    ├── log_patient                          │ │
│                       │    ├── get_patients_today                   │ │
│                       │    ├── get_patient_notes                    │ │
│                       │    ├── update_patient_note                  │ │
│                       │    ├── delete_patient_log                   │ │
│                       │    ├── set_reminder                         │ │
│                       │    └── summarise_shift                      │ │
│                       │         │                                    │ │
│                       │         ▼                                    │ │
│                       │  SQLite Database (data/neodoc.db)           │ │
│                       │  Session Logs (logs/*.json + *.txt)         │ │
│                       └──────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
```

**Data Flow for a Voice Interaction:**

1. The microphone captures audio via `sounddevice`.
2. `SpeechListener` performs VAD — it begins recording when energy exceeds the RMS threshold and stops after 1.5 s of silence.
3. The audio is transcribed by the Faster-Whisper model to text.
4. The text is sent to the local Ollama LLM with the conversation history and tool definitions.
5. The LLM returns either a direct response or a **tool call** (structured function invocation).
6. If a tool call is returned, the appropriate handler executes (e.g., inserts a row into SQLite).
   - For write operations (`log_patient`), an **approval gate** is triggered — the system reads back the captured details via TTS and waits for a voice "yes" or "no" before committing.
7. The tool result is fed back to the LLM for a natural-language final reply.
8. The final reply is queued to `SpeechQueue` and played via Piper TTS.
9. Throughout, `DisplayController` updates the OLED animation to reflect the current state.

---

## 4. Implementation Details

### 4.1 Main Application (`main.py`)

The main module (~1,250 lines) orchestrates the full voice interaction pipeline.

#### Configuration

```python
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
```

#### System Prompt

The LLM's behaviour is constrained by a clinical system prompt:

```python
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
  1. ALWAYS call the appropriate tool when the request matches a tool's purpose.
  2. After receiving a tool result, give a DETAILED, clinical-quality response.
  3. If a log entry was just created, always confirm the patient name, note, time, date, and entry ID.
  ...
"""
```

#### Voice Activity Detection (VAD)

The `SpeechListener` class implements energy-based VAD before transcription:

```python
def _record_utterance(self) -> list:
    chunks          = []
    speech_detected = False
    silence_frames  = 0

    silence_limit = int(_STT_SILENCE_SECONDS * self._sample_rate / self._chunk_frames)
    max_frames    = _STT_MAX_SECONDS * self._sample_rate

    with sd.InputStream(samplerate=self._sample_rate, channels=1,
                        dtype="float32", blocksize=self._chunk_frames) as stream:
        while total_frames < max_frames:
            data, _ = stream.read(self._chunk_frames)
            rms = float(np.sqrt(np.mean(data ** 2)))

            if rms >= _STT_ENERGY_THRESHOLD:
                speech_detected = True
                silence_frames  = 0
                chunks.append(data.copy())
            elif speech_detected:
                chunks.append(data.copy())
                silence_frames += 1
                if silence_frames >= silence_limit:
                    break
    return chunks
```

#### Voice Approval Gate

Before any patient log is committed to the database, the system reads the captured details back via TTS and waits for explicit confirmation:

```python
def _confirm_voice(listener, speech=None, record=None) -> bool:
    spoken_prompt = (
        f"I heard: Patient {record['patient_name']}. "
        f"Note: {record['note']}. "
        f"Time: {record['time']}. "
        f"Say yes to save this entry, or no to cancel."
    )

    speech.say(spoken_prompt)
    speech.wait()   # blocks until TTS playback finishes before opening mic

    for attempt in range(3):
        heard = listener.listen()
        if heard:
            if any(w in heard.lower() for w in ("yes", "confirm", "correct", "approved")):
                return True
            if any(w in heard.lower() for w in ("no", "cancel", "wrong", "discard")):
                return False
    return False    # default cancel after 3 unrecognised attempts
```

#### Database Schema

Patient data is persisted in a local SQLite database with two tables:

```python
# patient_logs table
CREATE TABLE IF NOT EXISTS patient_logs (
    id TEXT PRIMARY KEY,         -- UUID
    patient_name TEXT,
    note TEXT,
    time TEXT,
    date TEXT,
    logged_at TEXT               -- ISO 8601 timestamp
)

# reminders table
CREATE TABLE IF NOT EXISTS reminders (
    id TEXT PRIMARY KEY,
    message TEXT,
    delay_minutes INTEGER,
    created_at TEXT,
    trigger_at TEXT,
    completed INTEGER
)
```

#### Session Logging

Every interaction session is saved to `logs/` as both a JSON file (machine-readable) and a plain-text summary for audit purposes.

---

### 4.2 OLED Display Controller (`display.py`)

The `DisplayController` class drives a 128×64 SSD1306 OLED display over I2C, running its animation loop in a daemon background thread at ~20 FPS. It exposes a single method `set_state(state)` that `main.py` calls at key points in the interaction pipeline.

**Four animated states:**

| State | Visual | Triggered when |
|---|---|---|
| `idle` | Sleeping face with breathing animation | App starts / waiting for input |
| `listening` | Animated waveform bars | Microphone is open |
| `processing` | Spinning dot loader | LLM inference in progress |
| `speaking` | Talking face with expanding sound waves | TTS audio is playing |

```python
class DisplayController:
    def __init__(self):
        self.i2c  = busio.I2C(board.SCL, board.SDA)
        self.oled = adafruit_ssd1306.SSD1306_I2C(128, 64, self.i2c)
        self.state = "idle"
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self.frame_count = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def set_state(self, new_state):
        with self._lock:
            if self.state != new_state:
                self.state = new_state
                self.frame_count = 0   # reset animation on state change

    def _run_loop(self):
        while self._running:
            image = Image.new("1", (self.width, self.height))
            draw  = ImageDraw.Draw(image)
            with self._lock:
                current_state = self.state
                frame = self.frame_count
                self.frame_count += 1
            # dispatch to state-specific draw methods ...
            self.oled.image(image)
            self.oled.show()
            time.sleep(0.05)    # ~20 FPS
```

**Idle animation** (breathing sleeping face):

```python
def _draw_idle(self, draw, frame):
    cx, cy = self.width // 2, self.height // 2 - 5
    offset = int(math.sin(frame * 0.1) * 3)    # breathing effect
    draw.line((cx-25, cy-5+offset, cx-15, cy-5+offset), fill=255, width=2)  # left eye
    draw.line((cx+15, cy-5+offset, cx+25, cy-5+offset), fill=255, width=2)  # right eye
    draw.line((cx-5, cy+10+offset, cx+5, cy+10+offset), fill=255, width=2)  # mouth
    draw.text((2, self.height-12), "Idle", fill=255)
```

**Listening animation** (dynamic waveform bars):

```python
def _draw_listening(self, draw, frame):
    for i in range(7):                                     # 7 bars
        bh = 10 + abs(math.sin(frame * 0.2 + i)) * 20 + random.randint(0, 10)
        x  = start_x + i * (bar_width + spacing)
        draw.rectangle((x, cy - bh//2, x + bar_width, cy + bh//2), fill=255)
    draw.text((2, self.height-12), "Listening...", fill=255)
```

---

### 4.3 Tool System (`tools.json`)

The LLM's capabilities are defined as OpenAI-compatible function definitions. The model decides at runtime which tool (if any) to call, and with which arguments.

**Available tools:**

| Tool | Purpose |
|---|---|
| `log_patient` | Record a patient visit / clinical note |
| `get_patients_today` | Retrieve patient list for a shift/date |
| `get_patient_notes` | Look up all notes for a specific patient |
| `update_patient_note` | Amend a previously logged note |
| `delete_patient_log` | Remove an incorrect log entry by UUID |
| `set_reminder` | Set a timed alert in N minutes |
| `summarise_shift` | Generate brief / detailed / handover summary |

Example tool definition:

```json
{
  "type": "function",
  "function": {
    "name": "log_patient",
    "description": "Log a patient visit or clinical note ...",
    "parameters": {
      "type": "object",
      "properties": {
        "patient_name": { "type": "string", "description": "Name or identifier of the patient" },
        "note":         { "type": "string", "description": "The full clinical note to record" },
        "time":         { "type": "string", "description": "Time of the observation, e.g. '14:30'" }
      },
      "required": ["patient_name", "note"]
    }
  }
}
```

---

## 5. AI Implementation

### 5.1 Language Model (LLM)

**Model:** IBM Granite 4 (`granite4:350m`) — a 350-million parameter instruction-tuned language model with native tool-calling support, served via **Ollama**.

- Chosen for its small footprint (fits comfortably in ~1 GB RAM) while retaining reliable function-calling capability.
- Alternative: `qwen3.5:0.8b` tested as a fallback with slightly higher accuracy at the cost of more RAM.
- The model runs fully locally; Ollama exposes a REST API on `localhost:11434`.

**Tool-calling pipeline:**

```
User Prompt
    │
    ▼
LLM (First call) ──→ Direct Response (no tool)  →  TTS → Done
    │
    ▼ (tool call detected)
Tool Executor (local Python function)
    │
    ▼
LLM (Second call with tool result) → Natural Language Reply → TTS → Done
```

Streaming is enabled for real-time TTS: as the LLM generates tokens, the system splits the stream at sentence boundaries and feeds segments to the TTS queue immediately, so audio playback begins before the full response is complete.

### 5.2 Speech-to-Text (STT)

**Model:** Faster-Whisper `base` (OpenAI Whisper architecture, CTranslate2 backend)

- Quantised to `int8` for CPU inference: `WhisperModel(model_size, device="cpu", compute_type="int8")`
- Input: 16 kHz mono float32 PCM audio
- Language locked to English; `vad_filter=True` enabled for a second-pass silence filter
- Hallucination filter: common Whisper artefacts (e.g., "Thank you.", lone punctuation) are discarded via regex before returning the transcript

```python
_HALLUCINATION_RE = re.compile(
    r"^\s*(?:thank you\.?|thanks\.?|you\.?|\.+|,+)\s*$",
    re.IGNORECASE,
)
if not text or _HALLUCINATION_RE.match(text):
    return None
```

### 5.3 Text-to-Speech (TTS)

**Model:** Piper TTS (`en_US-lessac-medium.onnx`)

- Fully offline, runs on CPU, low latency
- Synthesises to a temporary WAV file, then plays via `pw-play` / `paplay` / `aplay` (auto-detected)
- TTS is decoupled from the main thread via `SpeechQueue` — a producer/consumer queue with a background daemon thread — so the main interaction loop is never blocked waiting for audio to finish playing

```python
class SpeechQueue:
    def say(self, text: str) -> None:
        cleaned = re.sub(r'\[.*?\]', '', text or "").strip()  # strip technical tags
        self._queue.put(cleaned)

    def wait(self) -> None:
        """Wait for all pending speech to finish before opening the microphone."""
        self._queue.join()
```

### 5.4 Edge Deployment

All three AI models run **on-device** with no network connectivity required after initial setup.

| Model | Size on Disk | RAM Usage (approx.) | Inference Target |
|---|---|---|---|
| Granite 4 350M (4-bit) | ~300 MB | ~600 MB | < 2 s / response |
| Faster-Whisper base (int8) | ~74 MB | ~200 MB | < 1 s / utterance |
| Piper TTS lessac-medium | ~63 MB | ~100 MB | < 0.5 s / sentence |

**Total estimated RAM footprint:** ~1 GB — feasible on a Raspberry Pi 4 (4 GB or 8 GB variant).

Ollama handles model loading, quantisation, and inference scheduling. Models are pulled once:

```bash
ollama pull granite4:350m
```

---

## 6. Hardware Implementation

### 6.1 Microcontroller / SBC

The system runs on a **Raspberry Pi** single-board computer. The full Linux environment supports Python 3, Ollama (ARM-compatible), and all required libraries.

> 📷 **[INSERT PHOTO: Raspberry Pi board — top-down view showing the board with connected peripherals]**

### 6.2 OLED Display

- **Model:** SSD1306 0.96" OLED (128×64 pixels, monochrome)
- **Interface:** I2C (SDA → Pin 3, SCL → Pin 5 on the GPIO header)
- **Driver library:** `adafruit-circuitpython-ssd1306`
- **Rendering library:** Pillow (`PIL.Image`, `PIL.ImageDraw`)

The display is controlled by `display.py` and updated at ~20 FPS in a separate thread. It provides four animated states corresponding to the AI's operating mode.

> 📷 **[INSERT PHOTO: OLED display showing the "idle" sleeping face animation]**

> 📷 **[INSERT PHOTO: OLED display showing the "listening" waveform animation]**

> 📷 **[INSERT PHOTO: OLED display showing the "processing" spinner animation]**

> 📷 **[INSERT PHOTO: OLED display showing the "speaking" talking face animation]**

### 6.3 Microphone Input

- **Device:** USB microphone or compatible audio input device
- Captured at **16 kHz, mono, float32** via the `sounddevice` library
- Sample rate verified at startup; automatic resampling to 16 kHz if the hardware uses a different native rate

> 📷 **[INSERT PHOTO: USB microphone connected to the Raspberry Pi]**

### 6.4 Audio Output

- **Device:** Bluetooth speaker or 3.5mm audio output
- Audio is played via system audio commands (`pw-play`, `paplay`, or `aplay`), automatically selected at runtime
- The `NEODOC_AUDIO_DEVICE` environment variable allows specifying a custom ALSA device

> 📷 **[INSERT PHOTO: Bluetooth speaker or audio output device in use]**

### 6.5 Physical Button (GPIO)

A physical push-button is connected to **GPIO Pin 7** to allow hardware-level interaction (e.g., triggering the display toggle).

- Pull-up/pull-down resistor configuration is handled in `button.py`
- Used as an alternative input mechanism without requiring speech

> 📷 **[INSERT PHOTO: Push-button wired to the GPIO header with breadboard connections]**

### Full Hardware Setup

> 📷 **[INSERT PHOTO: Full assembled hardware setup — Raspberry Pi, OLED display, microphone, speaker, and button]**

> 📷 **[INSERT PHOTO: Close-up of wiring and GPIO connections]**

---

## 7. Runtime Screenshots & Output

### 7.1 Application Startup

When the system starts, it loads the LLM, STT, and TTS models and initialises the OLED display:

> 📷 **[INSERT SCREENSHOT: Terminal output showing model loading messages, e.g., "[STT] Loading Whisper 'base' model … ready." and "[OLED] Display initialised."]**

### 7.2 Voice Interaction — Logging a Patient

Sample session: The nurse says *"Log John — high temperature, 39°C, given paracetamol"*

```
[STT] Listening … (speak now)
[STT] Processing …
User: Log John — high temperature, 39°C, given paracetamol

  ⚙  log_patient  args={'patient_name': 'John', 'note': 'High temperature 39°C, administered paracetamol', 'time': '14:32'}

  ┌─ APPROVAL REQUIRED ─────────────────────────────────┐
  │  Patient : John
  │  Note    : High temperature 39°C, administered paracetamol
  │  Time    : 14:32  |  Date: 2026-05-15
  └─────────────────────────────────────────────────────┘

[TTS PROMPT] I heard: Patient John. Note: High temperature 39°C, administered paracetamol. Time: 14:32. Say yes to save this entry, or no to cancel.

[STT] Listening … (speak now)
→ [LOGGED] Patient 'John' saved at 14:32 on 2026-05-15. Note: High temperature 39°C, administered paracetamol (id: 9719ffb2-eff3-4584-bc85-437d45564c0d)
```