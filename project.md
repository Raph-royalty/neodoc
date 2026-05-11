# 🏥 Locally-Deployed Conversational AI Assistant for Clinical Healthcare Workers

## 📌 Project Overview

This project aims to design and implement a **fully offline, voice-first AI assistant** for healthcare professionals (nurses/doctors) operating in clinical environments.

The assistant will:
- Capture voice input
- Understand intent using an LLM
- Perform actions (log, retrieve, remind)
- Respond using speech

All processing is **local (edge-based)** to ensure:
- Patient data privacy
- Reliability without internet
- Low latency interactions

---

## 👤 Target Users

### Primary Users
- Nurses
- Doctors

### Environment
- Hospitals, wards, clinics
- Low-connectivity or restricted internet zones

### Key Constraints
- Hands often occupied → voice-first interaction
- Time-sensitive workflows
- Sensitive data → must remain local

---

## 🎯 Core Features

### 1. Voice-Based Patient Logging
**Example:**
> "Log that Mary came in with fever at 3pm"

- Convert speech → text
- Extract:
  - Patient name
  - Symptoms
  - Timestamp
- Store in database

---

### 2. Record Retrieval
**Example:**
> "Who did I see this morning?"

- Query database
- Filter by time range
- Return summarized results

---

### 3. Reminder System
**Example:**
> "Remind me to check patient in bed 4 in 2 hours"

- Parse time intent
- Schedule reminder
- Trigger alert (voice/audio)

---

### 4. Drug/Dosage Query (RAG)
**Example:**
> "What's the dose of amoxicillin for a child?"

- Query local medical documents
- Return safe, referenced answer

---

### 5. Shift Summary Generation
**Example:**
> "Summarise my shift"

- Retrieve all logs
- Generate concise report

---

## 🧠 System Architecture

### 3-Layer Model

#### 1. Perception Layer
Handles input understanding

- Microphone input
- Speech-to-Text (STT)

**Tools:**
- Whisper (local)

---

#### 2. Reasoning Layer
Determines intent and action

- LLM interprets command
- Decides which tool to call

**Tools:**
- Ollama
- Phi-3 Mini

---

#### 3. Action + Response Layer

- Executes operations
- Generates response
- Converts to speech

**Tools:**
- SQLite (data)
- APScheduler (reminders)
- Piper TTS (speech output)

---

## 🧩 System Components

### 1. Speech-to-Text (STT)
- **Tool:** Whisper
- Input: Audio
- Output: Text

---

### 2. LLM Engine
- **Tool:** Ollama + Phi-3 Mini
- Role:
  - Intent classification
  - Tool calling
  - Response generation

---

### 3. Tool Layer (Core Logic)

#### a. Logging Tool
- Inserts records into SQLite

#### b. Query Tool
- Retrieves and filters records

#### c. Reminder Tool
- Schedules tasks using APScheduler

#### d. RAG Tool
- Retrieves medical knowledge

---

### 4. Memory System

#### Short-Term Memory
- Python list (conversation context)

#### Long-Term Memory
- SQLite database

#### Scheduled Memory
- APScheduler jobs

---

### 5. Text-to-Speech (TTS)
- **Tool:** Piper
- Output: Natural voice response

---

## 🗄️ Database Design (SQLite)

### Table: patients_logs
| Column        | Type    |
|--------------|--------|
| id           | INTEGER |
| patient_name | TEXT    |
| symptoms     | TEXT    |
| timestamp    | DATETIME |

---

### Table: reminders
| Column     | Type    |
|-----------|--------|
| id        | INTEGER |
| message   | TEXT    |
| trigger_at| DATETIME |

---

### Table: conversation_logs (optional)
| Column   | Type |
|----------|------|
| id       | INTEGER |
| input    | TEXT |
| response | TEXT |
| timestamp| DATETIME |

---


## ⚙️ Hardware Requirements

### Target Device
- Raspberry Pi 4B 4gb ram

### Peripherals
- USB Microphone
- Speaker or headphones

---

## 🧪 Software Stack

| Layer        | Tool |
|-------------|------|
| STT         | Whisper |
| LLM         | Ollama + Phi-3 Mini |
| DB          | SQLite |
| Scheduler   | APScheduler |
| TTS         | Piper |
| Backend     | Python |

---

## 🔄 System Flow

1. User speaks
2. Whisper converts to text
3. Text sent to LLM
4. LLM determines:
   - Intent
   - Tool to call
5. Tool executes
6. Response generated
7. Piper speaks response

---

## 🧠 Prompt Engineering

### System Prompt Example