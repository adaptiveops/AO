import lgpio
import time

# Define the I2C address and bus number
I2C_BUS = 1
I2C_ADDRESS = 0x40

# PCA9685 Registers
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06
LED0_OFF_L = 0x08

# Set PWM frequency to 50 Hz
prescale_value = int(25000000.0 / (4096 * 50) - 1)

try:
    h = lgpio.i2c_open(I2C_BUS, I2C_ADDRESS)
    print(f"Successfully opened I2C connection to address 0x{I2C_ADDRESS:X}")

    # Set PCA9685 mode
    lgpio.i2c_write_device(h, [MODE1, 0x00])
    time.sleep(0.1)
    lgpio.i2c_write_device(h, [MODE1, 0x10])
    time.sleep(0.1)
    lgpio.i2c_write_device(h, [PRESCALE, prescale_value])
    time.sleep(0.1)
    lgpio.i2c_write_device(h, [MODE1, 0x00])
    time.sleep(0.1)
    lgpio.i2c_write_device(h, [MODE1, 0xa1])
    time.sleep(0.1)

    # Set PWM for channel 0 (example values)
    lgpio.i2c_write_device(h, [LED0_ON_L, 0x00, 0x00])
    lgpio.i2c_write_device(h, [LED0_OFF_L, 0x10, 0x00])

    lgpio.i2c_close(h)
except lgpio.error as e:
    print(f"I2C communication error: {e}")
except Exception as e:
    print(f"Failed to open I2C connection: {e}")
