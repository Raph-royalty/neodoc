import os
os.environ["JETSON_MODEL_NAME"] = "JETSON_ORIN_NANO"
import board
import busio
import time
import threading
import math
import random
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306

class DisplayController:
    def __init__(self):
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.oled = adafruit_ssd1306.SSD1306_I2C(128, 64, self.i2c)
        self.width = self.oled.width
        self.height = self.oled.height
        
        self.state = "idle"
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        
        self.frame_count = 0

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=0.5)
        finally:
            try:
                self.oled.fill(0)
                self.oled.show()
            except Exception:
                pass

    def set_state(self, new_state):
        with self._lock:
            if self.state != new_state:
                self.state = new_state
                self.frame_count = 0 # reset animation on state change

    def _run_loop(self):
        while self._running:
            image = Image.new("1", (self.width, self.height))
            draw = ImageDraw.Draw(image)
            
            with self._lock:
                current_state = self.state
                frame = self.frame_count
                self.frame_count += 1
                
            if current_state == "idle":
                self._draw_idle(draw, frame)
            elif current_state == "listening":
                self._draw_listening(draw, frame)
            elif current_state == "processing":
                self._draw_processing(draw, frame)
            elif current_state == "speaking":
                self._draw_speaking(draw, frame)
                
            self.oled.image(image)
            self.oled.show()
            time.sleep(0.05) # ~20 FPS

    def _draw_idle(self, draw, frame):
        # Draw a sleeping/resting face
        cx, cy = self.width // 2, self.height // 2 - 5
        
        # Breathing effect
        offset = int(math.sin(frame * 0.1) * 3)
        
        # Eyes (- -)
        draw.line((cx - 25, cy - 5 + offset, cx - 15, cy - 5 + offset), fill=255, width=2)
        draw.line((cx + 15, cy - 5 + offset, cx + 25, cy - 5 + offset), fill=255, width=2)
        
        # Mouth _
        draw.line((cx - 5, cy + 10 + offset, cx + 5, cy + 10 + offset), fill=255, width=2)
        
        draw.text((2, self.height - 12), "Idle", fill=255)

    def _draw_listening(self, draw, frame):
        # Draw a moving waveform
        cx, cy = self.width // 2, self.height // 2 - 5
        num_bars = 7
        bar_width = 6
        spacing = 4
        total_width = num_bars * bar_width + (num_bars - 1) * spacing
        start_x = cx - total_width // 2
        
        for i in range(num_bars):
            # Base height + random fluctuation + sine wave
            bh = 10 + abs(math.sin(frame * 0.2 + i)) * 20 + random.randint(0, 10)
            x = start_x + i * (bar_width + spacing)
            draw.rectangle((x, cy - bh//2, x + bar_width, cy + bh//2), fill=255)
            
        draw.text((2, self.height - 12), "Listening...", fill=255)

    def _draw_processing(self, draw, frame):
        # Draw a spinner / loading circles
        cx, cy = self.width // 2, self.height // 2 - 5
        
        num_dots = 8
        radius = 15
        
        for i in range(num_dots):
            angle = math.pi * 2 * i / num_dots
            # highlight one dot based on frame
            active = (frame // 2) % num_dots == i
            r = 4 if active else 2
            
            px = cx + int(math.cos(angle) * radius)
            py = cy + int(math.sin(angle) * radius)
            
            draw.ellipse((px-r, py-r, px+r, py+r), fill=255)
            
        draw.text((2, self.height - 12), "Processing...", fill=255)

    def _draw_speaking(self, draw, frame):
        # Draw a talking face with expanding sound waves
        cx, cy = self.width // 2, self.height // 2 - 5
        
        # Eyes (O O)
        draw.ellipse((cx - 20, cy - 10, cx - 12, cy - 2), fill=255)
        draw.ellipse((cx + 12, cy - 10, cx + 20, cy - 2), fill=255)
        
        # Mouth moving
        mouth_h = int(5 + abs(math.sin(frame * 0.4)) * 10)
        draw.ellipse((cx - 8, cy + 5, cx + 8, cy + 5 + mouth_h), fill=255)
        
        # Sound waves
        wave_radius = (frame * 2) % 30
        draw.arc((cx - 30 - wave_radius, cy - 20 - wave_radius, 
                  cx + 30 + wave_radius, cy + 20 + wave_radius), 
                 start=150, end=210, fill=255, width=2)
        draw.arc((cx - 30 - wave_radius, cy - 20 - wave_radius, 
                  cx + 30 + wave_radius, cy + 20 + wave_radius), 
                 start=-30, end=30, fill=255, width=2)

        draw.text((2, self.height - 12), "Speaking...", fill=255)
