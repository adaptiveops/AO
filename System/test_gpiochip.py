import lgpio
import time

# Set up a test servo
test_pin = 12
min_pulse = 1000  # in microseconds
max_pulse = 2000  # in microseconds

def set_servo_pulse(chip, pin, pulse_width):
    pulse_width_us = pulse_width
    try:
        print(f"Setting PWM on gpiochip {chip} pin {pin} to {pulse_width_us} us")
        lgpio.tx_pwm(chip, pin, 50, pulse_width_us)  # 50 Hz frequency
    except lgpio.error as e:
        print(f"Failed to set PWM on gpiochip {chip} pin {pin}: {e}")

for chip_num in range(5):
    try:
        print(f"Trying gpiochip{chip_num}")
        chip = lgpio.gpiochip_open(chip_num)
        set_servo_pulse(chip, test_pin, 1500)  # middle position (1500us)
        time.sleep(1)
        lgpio.gpiochip_close(chip)
        print(f"gpiochip{chip_num} works.")
    except lgpio.error as e:
        print(f"gpiochip{chip_num} failed: {e}")
