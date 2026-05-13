import os
import sys
import wave
import subprocess
import shutil
from pathlib import Path

try:
    from piper.voice import PiperVoice
except ImportError:
    print("Error: Piper not installed. Run: pip install piper-tts")
    sys.exit(1)

VOICE_PATH = Path("voices") / "en_US-lessac-medium.onnx"
OUTPUT_WAV = Path("test_output.wav")
TEXT_TO_SPEAK = "Hello! This is a test of the Piper text to speech system."


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode, out.strip()
    except FileNotFoundError:
        return 127, "command not found"


def print_audio_diagnostics():
    print("\n[diag] Python:", sys.version.split()[0])
    print("[diag] Platform:", sys.platform)
    print("[diag] NEODOC_AUDIO_DEVICE:", os.getenv("NEODOC_AUDIO_DEVICE") or "(unset)")
    print("[diag] PULSE_SERVER:", os.getenv("PULSE_SERVER") or "(unset)")
    print("[diag] PIPEWIRE_REMOTE:", os.getenv("PIPEWIRE_REMOTE") or "(unset)")

    candidates = [
        "pw-play",
        "paplay",
        "aplay",
        "play",
        "ffplay",
        "pactl",
        "wpctl",
    ]
    available = {c: (shutil.which(c) or "") for c in candidates}
    print("[diag] Binaries:")
    for c in candidates:
        print(f"  - {c}: {available[c] or '(missing)'}")

    if available["wpctl"]:
        code, out = run_cmd([available["wpctl"], "status"])
        print("\n[diag] wpctl status (PipeWire):")
        print(out if out else f"(exit {code})")

    if available["pactl"]:
        code, out = run_cmd([available["pactl"], "info"])
        print("\n[diag] pactl info (PulseAudio/PipeWire-Pulse):")
        print(out if out else f"(exit {code})")
        code, out = run_cmd([available["pactl"], "list", "short", "sinks"])
        print("\n[diag] pactl list short sinks:")
        print(out if out else f"(exit {code})")

    if available["aplay"]:
        code, out = run_cmd([available["aplay"], "-l"])
        print("\n[diag] aplay -l (hardware cards):")
        print(out if out else f"(exit {code})")
        code, out = run_cmd([available["aplay"], "-L"])
        print("\n[diag] aplay -L (PCM devices):")
        # This can be long; trim a bit.
        if out:
            lines = out.splitlines()
            print("\n".join(lines[:80]) + ("\n... (trimmed)" if len(lines) > 80 else ""))
        else:
            print(f"(exit {code})")

def test_synthesis():
    if not VOICE_PATH.exists():
        print(f"Error: Voice model not found at {VOICE_PATH}")
        return False

    print(f"[*] Loading voice from {VOICE_PATH}...")
    voice = PiperVoice.load(str(VOICE_PATH))
    print(f"[*] Voice sample rate: {voice.config.sample_rate} Hz")
    
    print(f"[*] Synthesizing audio to {OUTPUT_WAV}...")
    with wave.open(str(OUTPUT_WAV), "wb") as wav_file:
        # Newer piper-tts: use synthesize_wav (writes frames to wave writer)
        if hasattr(voice, "synthesize_wav"):
            voice.synthesize_wav(TEXT_TO_SPEAK, wav_file)
        else:
            # Older piper-tts: synthesize(text, wav_file)
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(voice.config.sample_rate)
            voice.synthesize(TEXT_TO_SPEAK, wav_file)
    
    size = OUTPUT_WAV.stat().st_size
    print(f"[*] Audio synthesized successfully. File size: {size} bytes")
    if size < 1000:
        print("[!] Warning: The generated WAV file is very small. Synthesis may have failed.")
    try:
        with wave.open(str(OUTPUT_WAV), "rb") as r:
            duration = r.getnframes() / float(r.getframerate() or 1)
            print(
                f"[*] WAV info: channels={r.getnchannels()} "
                f"width={r.getsampwidth()*8}-bit rate={r.getframerate()}Hz "
                f"frames={r.getnframes()} duration={duration:.2f}s"
            )
            if r.getnframes() == 0:
                print("[!] WAV has 0 frames; synthesis produced no audio samples.")
    except Exception as e:
        print(f"[!] Could not re-open WAV for inspection: {e}")
    return True

def test_playback():
    print(f"\n[*] Beginning playback diagnosis...")

    print_audio_diagnostics()

    # On modern Linux desktops, PipeWire/Pulse paths are usually correct.
    # ALSA-only playback (aplay + hw/plughw) can be silent even when desktop audio works.
    players = ["pw-play", "paplay", "aplay", "play", "ffplay"]
    
    for player in players:
        player_path = shutil.which(player)
        if player_path:
            print(f"\n>>> Trying playback with {player_path}...")
            cmd = [player_path, str(OUTPUT_WAV)]
            
            # If using aplay, try *without* and then *with* the ALSA override.
            if player == "aplay":
                cmd_variants = [[player_path, "-q", str(OUTPUT_WAV)]]
                device = os.getenv("NEODOC_AUDIO_DEVICE")
                if device:
                    cmd_variants.append([player_path, "-q", "-D", device, str(OUTPUT_WAV)])
            else:
                cmd_variants = [cmd]
            
            try:
                for idx, one_cmd in enumerate(cmd_variants, 1):
                    print("    $", " ".join(one_cmd))
                    subprocess.run(one_cmd, check=True)
                    suffix = "" if len(cmd_variants) == 1 else f" (variant {idx}/{len(cmd_variants)})"
                    print(f"[?] Playback with {player}{suffix} finished. Did you hear it?")
                    ans = input("Did you hear audio? [y/N]: ").strip().lower()
                    if ans == "y":
                        print("\n[*] Excellent! We identified a working player.")
                        print(f"    Recommended Linux player for main.py: '{player}'.")
                        if player == "aplay":
                            device = os.getenv("NEODOC_AUDIO_DEVICE")
                            if device:
                                print(f"    Your NEODOC_AUDIO_DEVICE override worked: {device}")
                            else:
                                print("    Use default ALSA device; avoid forcing plughw unless needed.")
                        return
            except subprocess.CalledProcessError as e:
                print(f"[!] Playback with {player} failed with exit code {e.returncode}: {e}")
            except Exception as e:
                print(f"[!] Playback with {player} failed: {e}")
        else:
            print(f"[-] Player '{player}' not found on system.")
            
    print("\n[!] All automatic playback methods exhausted.")
    print(f"    You can try manually playing {OUTPUT_WAV} with your system's default media player to verify if the file itself is valid.")

if __name__ == "__main__":
    if test_synthesis():
        test_playback()
