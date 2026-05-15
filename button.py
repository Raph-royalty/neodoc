import os
os.environ["JETSON_MODEL_NAME"] = "JETSON_ORIN_NANO"
import board
import busio
import time
import digitalio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306

# Initialize I2C and OLED
i2c = busio.I2C(board.SCL, board.SDA)
oled = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c)

# Initialize Button
button = digitalio.DigitalInOut(board.D4)
button.direction = digitalio.Direction.INPUT
button.pull = digitalio.Pull.UP

# State tracking
display_on = False
last_button_state = True

def show_message():
    """Display message on OLED"""
    image = Image.new("1", (oled.width, oled.height))
    draw = ImageDraw.Draw(image)
    draw.text((10, 25), "Hello Raphael", fill=255)
    oled.image(image)
    oled.show()

def clear_display():
    """Clear the OLED"""
    oled.fill(0)
    oled.show()

print("Press button to toggle display!")
print("Press Ctrl+C to exit")

try:
    while True:
        current_button_state = button.value
        
        # Detect button press (transition from True to False)
        if last_button_state and not current_button_state:
            print("Button pressed - toggling display")
            
            # Toggle display state
            display_on = not display_on
            
            if display_on:
                show_message()
                print("Display ON")
            else:
                clear_display()
                print("Display OFF")
            
            # Wait for button release
            while not button.value:
                time.sleep(0.01)
            
            time.sleep(0.2)  # Debounce delay
        
        last_button_state = current_button_state
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nCleaning up...")
    clear_display()
    button.deinit()