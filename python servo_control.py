import time
import lgpio

# Define the I2C address and bus number
I2C_BUS = 1
I2C_ADDRESS = 0x40

# PCA9685 Registers
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06
LED0_OFF_L = 0x08

# Initialize the I2C connection using lgpio
try:
    h = lgpio.i2c_open(I2C_BUS, I2C_ADDRESS)
    print(f"Successfully opened I2C connection to address 0x{I2C_ADDRESS:X}")
except Exception as e:
    print(f"Failed to open I2C connection: {e}")
    exit(1)

def write_byte_data(handle, reg, value):
    try:
        data = bytes([reg, value])
        lgpio.i2c_write_device(handle, data)
        print(f"Write to reg 0x{reg:02X}: 0x{value:02X}")
    except lgpio.error as e:
        print(f"Failed to write byte data to reg 0x{reg:02X}: {e}")

def write_word_data(handle, reg, value):
    try:
        data = bytes([reg, value & 0xFF, (value >> 8) & 0xFF])
        lgpio.i2c_write_device(handle, data)
        print(f"Write to reg 0x{reg:02X}: 0x{value:04X}")
    except lgpio.error as e:
        print(f"Failed to write word data to reg 0x{reg:02X}: {e}")

# Set PCA9685 mode
write_byte_data(h, MODE1, 0x00)

# Set PWM frequency to 50 Hz
prescale_value = int(25000000.0 / (4096 * 50) - 1)
write_byte_data(h, MODE1, 0x10)  # Enter sleep mode
write_byte_data(h, PRESCALE, prescale_value)
write_byte_data(h, MODE1, 0x00)  # Wake up
time.sleep(0.005)
write_byte_data(h, MODE1, 0xa1)  # Restart and enable auto increment

# Function to set PWM for a channel
def set_pwm(channel, on, off):
    write_word_data(h, LED0_ON_L + 4 * channel, on)
    write_word_data(h, LED0_OFF_L + 4 * channel, off)

# Function to set the angle of the servo
def set_servo_angle(channel, angle):
    pulse_length = 4096
    pulse = int((angle / 180.0) * pulse_length)
    set_pwm(channel, 0, pulse)

# Main loop to test the servo
try:
    while True:
        for angle in range(0, 180, 5):  # 0 to 180 degrees, step by 5 degrees
            set_servo_angle(0, angle)
            time.sleep(0.1)
        for angle in range(180, 0, -5):  # 180 to 0 degrees, step by -5 degrees
            set_servo_angle(0, angle)
            time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    # Cleanup
    lgpio.i2c_close(h)
