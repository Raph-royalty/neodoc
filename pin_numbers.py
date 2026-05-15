import os
os.environ["JETSON_MODEL_NAME"] = "JETSON_ORIN_NANO"

import Jetson.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)

import Jetson.GPIO.gpio_pin_data as pd
model, info, channel_data = pd.get_data()
for name, ch in channel_data.items():
    print(name, "→", ch)